
import os
import sys
import io
import csv
import threading
from datetime import datetime

import numpy as np
from scipy.signal import (
    butter, sosfiltfilt, iirnotch,
    filtfilt, find_peaks
)

import plotly.graph_objects as go

from dash import (
    Dash, html, dcc, Input, Output,
    State, ctx, no_update
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


FS = 250
BLOCK = 125
MAX_SECONDS = 300
VIEW_SECONDS = 10

BG = "#0b1422"
CARD = "#16263a"
CYAN = "#52d5eb"
WHITE = "#eaf4ff"
MUTED = "#9eb2c8"
GRID = "#2a4055"
GREEN = "#5bd5a3"
YELLOW = "#f5b65c"
RED = "#ff7587"

DAQ_TASK = None
DAQ_LOCK = threading.Lock()


def detener_daq():
    global DAQ_TASK

    with DAQ_LOCK:
        if DAQ_TASK is not None:
            try:
                DAQ_TASK.close()
            finally:
                DAQ_TASK = None


def iniciar_daq(canal):
    global DAQ_TASK

    if sys.platform != "win32":
        raise RuntimeError(
            "NI USB-6008 requiere Windows con NI-DAQmx."
        )

    try:
        import nidaqmx
        from nidaqmx.constants import (
            TerminalConfiguration,
            AcquisitionType
        )
    except ImportError:
        raise RuntimeError(
            "Instala nidaqmx y el controlador NI-DAQmx."
        )

    detener_daq()

    tarea = nidaqmx.Task()

    try:
        tarea.ai_channels.add_ai_voltage_chan(
            canal,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=-10.0,
            max_val=10.0
        )

        tarea.timing.cfg_samp_clk_timing(
            rate=FS,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=FS * 10
        )

        tarea.start()

        with DAQ_LOCK:
            DAQ_TASK = tarea

    except Exception:
        tarea.close()
        raise


def leer_daq():
    with DAQ_LOCK:
        if DAQ_TASK is None:
            raise RuntimeError("DAQ no iniciada.")

        valores = DAQ_TASK.read(
            number_of_samples_per_channel=BLOCK,
            timeout=3.0
        )

    return np.asarray(valores, dtype=float)


def simular_ecg(t):
    fase = np.mod(t, 0.8)

    p = 0.13 * np.exp(-((fase - 0.17) / 0.035)**2)
    q = -0.13 * np.exp(-((fase - 0.365) / 0.012)**2)
    r = 1.13 * np.exp(-((fase - 0.40) / 0.010)**2)
    s = -0.26 * np.exp(-((fase - 0.433) / 0.012)**2)
    onda_t = 0.30 * np.exp(-((fase - 0.63) / 0.07)**2)

    ruido = 0.012 * np.random.randn(len(t))
    red = 0.025 * np.sin(2 * np.pi * 50 * t)
    base = 0.035 * np.sin(2 * np.pi * 0.25 * t)

    return p + q + r + s + onda_t + ruido + red + base


def filtrar(y, modo="completo"):
    y = np.asarray(y, dtype=float)

    if len(y) < 100 or modo == "ninguno":
        return y.copy()

    resultado = y.copy()

    if modo in ["pasabanda", "completo"]:
        sos = butter(
            3, [0.5, 40],
            btype="bandpass",
            fs=FS,
            output="sos"
        )

        resultado = sosfiltfilt(sos, resultado)

    if modo in ["notch", "completo"]:
        b, a = iirnotch(50, Q=30, fs=FS)
        resultado = filtfilt(b, a, resultado)

    return resultado


def analizar_ecg(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    resultado = {
        "bpm": None,
        "rr": [],
        "rr_medio": None,
        "sdnn": None,
        "picos": [],
        "amplitud": None,
        "qrs": None,
        "cantidad": 0
    }

    if len(y) < FS * 4:
        return resultado

    if not np.all(np.isfinite(y)):
        return resultado

    señal = filtrar(y, "pasabanda")

    rango = (
        np.percentile(señal, 99) -
        np.percentile(señal, 20)
    )

    if rango < 0.03:
        return resultado

    picos, _ = find_peaks(
        señal,
        distance=int(0.45 * FS),
        prominence=max(0.06, 0.45 * rango)
    )

    resultado["picos"] = picos.tolist()
    resultado["cantidad"] = len(picos)

    amplitudes = []
    duraciones = []

    for p in picos:
        izquierda = max(0, p - int(0.20 * FS))
        derecha = min(len(señal), p + int(0.20 * FS))

        segmento = señal[izquierda:derecha]

        if len(segmento) < 15:
            continue

        amplitudes.append(
            float(señal[p] - np.min(segmento))
        )

        derivada = np.abs(np.diff(segmento))

        if len(derivada) < 5:
            continue

        centro = p - izquierda
        umbral = max(
            0.12 * np.max(derivada), 0.001
        )

        a = centro
        b = centro

        while a > 1:
            if derivada[a - 1] < umbral:
                break
            a -= 1

        while b < len(derivada) - 1:
            if derivada[b] < umbral:
                break
            b += 1

        duracion = (b - a) * 1000 / FS

        if 20 <= duracion <= 200:
            duraciones.append(duracion)

    if amplitudes:
        resultado["amplitud"] = float(
            np.median(amplitudes)
        )

    if duraciones:
        resultado["qrs"] = float(
            np.median(duraciones)
        )

    if len(picos) >= 3:
        rr = np.diff(t[picos]) * 1000

        rr = rr[(rr >= 330) & (rr <= 1800)]

        if len(rr):
            mediana = float(np.median(rr))
            bpm = 60000 / mediana

            resultado["rr"] = rr.tolist()
            resultado["rr_medio"] = mediana

            if 35 <= bpm <= 180:
                resultado["bpm"] = float(bpm)

            if len(rr) > 1:
                resultado["sdnn"] = float(
                    np.std(rr, ddof=1)
                )

    return resultado


def calidad_senal(y, voltajes, errores):
    y = np.asarray(y, dtype=float)
    voltajes = np.asarray(voltajes, dtype=float)

    if errores > 0:
        return "REVISAR ADQUISICIÓN"

    if len(y) < FS * 2:
        return "ESPERANDO DATOS"

    if not np.all(np.isfinite(y)):
        return "DATOS INVÁLIDOS"

    if len(voltajes):
        saturacion = np.mean(
            np.abs(voltajes) >= 9.8
        )

        if saturacion > 0.005:
            return "SATURACIÓN"

    if np.ptp(y) < 0.03:
        return "SEÑAL MUY BAJA"

    procesada = filtrar(y, "completo")
    ruido = y - procesada

    relacion = (
        np.std(ruido) /
        max(np.std(procesada), 0.0001)
    )

    if relacion > 0.6:
        return "RUIDO ELEVADO"

    return "SEÑAL ESTABLE"


def grafica(t, y, titulo, picos=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t,
        y=y,
        mode="lines",
        name=titulo,
        line=dict(color=CYAN, width=2)
    ))

    if picos is not None and len(picos):
        indices = np.asarray(picos, dtype=int)
        indices = indices[
            (indices >= 0) & (indices < len(y))
        ]

        fig.add_trace(go.Scatter(
            x=np.asarray(t)[indices],
            y=np.asarray(y)[indices],
            mode="markers",
            name="Picos R",
            marker=dict(color=YELLOW, size=9)
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=WHITE),
        height=360,
        margin=dict(l=50, r=20, t=25, b=45),
        xaxis=dict(
            title="Tiempo (s)",
            gridcolor=GRID
        ),
        yaxis=dict(
            title="Amplitud (mV)",
            gridcolor=GRID
        ),
        legend=dict(orientation="h", y=1.15),
        uirevision=titulo
    )

    return fig


def grafica_fft(y):
    fig = go.Figure()
    y = np.asarray(y)

    if len(y) >= FS:
        y = y - np.mean(y)

        espectro = np.abs(
            np.fft.rfft(y * np.hanning(len(y)))
        ) * 2 / len(y)

        frecuencias = np.fft.rfftfreq(
            len(y), 1 / FS
        )

        mascara = frecuencias <= 100

        fig.add_trace(go.Scatter(
            x=frecuencias[mascara],
            y=espectro[mascara],
            mode="lines",
            line=dict(color=YELLOW)
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=WHITE),
        height=310,
        margin=dict(l=50, r=20, t=20, b=45),
        xaxis=dict(
            title="Frecuencia (Hz)",
            gridcolor=GRID
        ),
        yaxis=dict(
            title="Magnitud",
            gridcolor=GRID
        )
    )

    return fig


def grafica_rr(rr):
    fig = go.Figure()

    if len(rr):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(rr) + 1)),
            y=rr,
            mode="lines+markers",
            line=dict(color=CYAN)
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=WHITE),
        height=300,
        margin=dict(l=50, r=20, t=20, b=45),
        xaxis=dict(
            title="Número de intervalo",
            gridcolor=GRID
        ),
        yaxis=dict(
            title="RR (ms)",
            gridcolor=GRID
        )
    )

    return fig


def panel(contenido):
    return html.Div(
        contenido,
        style={
            "background": CARD,
            "border": "1px solid " + GRID,
            "borderRadius": "15px",
            "padding": "20px",
            "marginBottom": "16px"
        }
    )


def tarjeta(titulo, identificador):
    return html.Div([
        html.Div(
            titulo,
            style={
                "color": MUTED,
                "fontSize": "13px"
            }
        ),
        html.Div(
            "--",
            id=identificador,
            style={
                "color": CYAN,
                "fontSize": "27px",
                "fontWeight": "bold",
                "marginTop": "8px"
            }
        )
    ], style={
        "background": CARD,
        "border": "1px solid " + GRID,
        "borderRadius": "12px",
        "padding": "18px",
        "flex": "1",
        "minWidth": "145px"
    })


def boton(texto, identificador, color=CYAN):
    return html.Button(
        texto,
        id=identificador,
        n_clicks=0,
        style={
            "background": color,
            "color": BG,
            "border": "none",
            "borderRadius": "8px",
            "padding": "12px 17px",
            "margin": "5px",
            "fontWeight": "bold",
            "cursor": "pointer"
        }
    )


def estadisticas(elementos):
    return html.Div([
        tarjeta(titulo, identificador)
        for titulo, identificador in elementos
    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "12px",
        "marginBottom": "16px"
    })


def selector(identificador, opciones, valor):
    return dcc.Dropdown(
        id=identificador,
        options=opciones,
        value=valor,
        clearable=False,
        style={
            "color": BG,
            "marginBottom": "15px"
        }
    )


def entrada(identificador, valor, tipo="text"):
    return dcc.Input(
        id=identificador,
        value=valor,
        type=tipo,
        style={
            "width": "100%",
            "maxWidth": "400px",
            "padding": "11px",
            "borderRadius": "8px",
            "border": "1px solid " + GRID,
            "marginBottom": "18px",
            "boxSizing": "border-box"
        }
    )


def etiqueta(texto):
    return html.Label(
        texto,
        style={
            "display": "block",
            "color": MUTED,
            "marginBottom": "7px"
        }
    )


app = Dash(__name__)
app.title = "ECG LAB | Monitor Hospitalario"


app.layout = html.Div([

    html.H1(
        "♥ ECG LAB",
        style={
            "color": CYAN,
            "marginBottom": "5px"
        }
    ),

    html.P(
        "SISTEMA DE MONITOREO Y ANÁLISIS ELECTROCARDIOGRÁFICO",
        style={"color": MUTED}
    ),

    dcc.Tabs(
        id="tabs",
        value="monitor",
        colors={
            "border": GRID,
            "primary": CYAN,
            "background": CARD
        },
        children=[

            dcc.Tab(
                label="Monitor ECG",
                value="monitor",
                children=[
                    html.Br(),

                    estadisticas([
                        ("Frecuencia cardíaca", "bpm"),
                        ("Amplitud R estimada", "amplitud"),
                        ("QRS estimado", "qrs"),
                        ("Calidad de señal", "calidad")
                    ]),

                    panel([
                        html.H3("Monitor cardíaco"),
                        dcc.Graph(id="ecg")
                    ]),

                    panel([
                        boton("INICIAR", "iniciar"),
                        boton("DETENER", "detener", YELLOW),
                        boton("DESCARGAR CSV", "csv"),
                        boton("GUARDAR SESIÓN", "guardar"),

                        html.P(
                            "",
                            id="mensaje",
                            style={"color": YELLOW}
                        ),

                        html.P(
                            "",
                            id="guardado",
                            style={"color": GREEN}
                        )
                    ])
                ]
            ),

            dcc.Tab(
                label="Procesamiento",
                value="procesamiento",
                children=[
                    html.Br(),

                    panel([
                        html.H3("Filtros digitales"),

                        selector(
                            "filtro",
                            [
                                {
                                    "label": "Sin filtro",
                                    "value": "ninguno"
                                },
                                {
                                    "label": "Pasabanda 0.5 - 40 Hz",
                                    "value": "pasabanda"
                                },
                                {
                                    "label": "Notch 50 Hz",
                                    "value": "notch"
                                },
                                {
                                    "label": "Pasabanda + Notch",
                                    "value": "completo"
                                }
                            ],
                            "completo"
                        )
                    ]),

                    panel([
                        html.H3("Señal original"),
                        dcc.Graph(id="original")
                    ]),

                    panel([
                        html.H3("Señal filtrada"),
                        dcc.Graph(id="filtrada")
                    ]),

                    panel([
                        html.H3("Espectro FFT"),
                        dcc.Graph(id="fft")
                    ])
                ]
            ),

            dcc.Tab(
                label="Análisis ECG",
                value="analisis",
                children=[
                    html.Br(),

                    estadisticas([
                        ("RR mediano", "rr"),
                        ("SDNN", "sdnn"),
                        ("Picos R detectados", "cantidad"),
                        ("Frecuencia cardíaca", "bpm2")
                    ]),

                    panel([
                        html.H3("Intervalos RR"),
                        dcc.Graph(id="rrgrafica")
                    ])
                ]
            ),

            dcc.Tab(
                label="Control DAQ",
                value="control",
                children=[
                    html.Br(),

                    estadisticas([
                        ("Muestras recibidas", "muestras"),
                        ("Pérdidas verificadas", "perdidas"),
                        ("Duración del registro", "duracion"),
                        ("Estado", "conexion")
                    ]),

                    panel([
                        html.H3("Adquisición"),
                        html.P("Frecuencia de muestreo: 250 Hz"),
                        html.P("Bloque: 125 muestras"),
                        html.P("Ventana de visualización: 10 s"),
                        html.P("Memoria de sesión: 5 minutos"),

                        html.H3("Errores detectados"),
                        html.H2(
                            "0",
                            id="errores",
                            style={"color": RED}
                        ),

                        html.P(
                            "La verificación exacta de muestras "
                            "perdidas no está implementada.",
                            style={"color": YELLOW}
                        )
                    ])
                ]
            ),

            dcc.Tab(
                label="Configuración",
                value="configuracion",
                children=[
                    html.Br(),

                    panel([
                        html.H3("Fuente de señal"),

                        selector(
                            "fuente",
                            [
                                {
                                    "label": "ECG simulado",
                                    "value": "simulado"
                                },
                                {
                                    "label": "NI USB-6008 (Windows)",
                                    "value": "daq"
                                }
                            ],
                            "simulado"
                        ),

                        etiqueta("Canal de adquisición"),
                        entrada("canal", "Dev1/ai0")
                    ]),

                    panel([
                        html.H3("Calibración DAQ"),

                        etiqueta("Ganancia del amplificador (V/V)"),
                        entrada("ganancia", 1000, "number"),

                        etiqueta("Offset de salida (V)"),
                        entrada("offset", 0, "number"),

                        etiqueta("Factor de escala adicional"),
                        entrada("escala", 1, "number"),

                        html.P(
                            "ECG (mV) = "
                            "(V_DAQ - offset) × 1000 "
                            "× escala / ganancia",
                            style={"color": CYAN}
                        ),

                        html.P(
                            "Los cambios se aplican "
                            "al iniciar una nueva sesión.",
                            style={"color": MUTED}
                        )
                    ])
                ]
            ),

            dcc.Tab(
                label="Informe ECG",
                value="informe",
                children=[
                    html.Br(),

                    panel([
                        html.H2(
                            "Datos del informe",
                            style={"color": CYAN}
                        ),

                        etiqueta("Nombre del paciente o código"),
                        entrada("paciente", ""),

                        etiqueta("Edad"),
                        entrada("edad", None, "number"),

                        etiqueta("Sexo"),
                        selector(
                            "sexo",
                            [
                                {
                                    "label": "No especificado",
                                    "value": "No especificado"
                                },
                                {
                                    "label": "Masculino",
                                    "value": "Masculino"
                                },
                                {
                                    "label": "Femenino",
                                    "value": "Femenino"
                                },
                                {
                                    "label": "Otro",
                                    "value": "Otro"
                                }
                            ],
                            "No especificado"
                        ),

                        etiqueta("Responsable de la medición"),
                        entrada("responsable", ""),

                        etiqueta("Derivación ECG"),
                        selector(
                            "derivacion",
                            [
                                {"label": "I", "value": "I"},
                                {"label": "II", "value": "II"},
                                {"label": "III", "value": "III"},
                                {"label": "Otra", "value": "Otra"},
                                {
                                    "label": "No especificada",
                                    "value": "No especificada"
                                }
                            ],
                            "No especificada"
                        ),

                        etiqueta("Observaciones"),

                        dcc.Textarea(
                            id="observaciones",
                            placeholder=(
                                "Escribe aquí las observaciones "
                                "de la medición..."
                            ),
                            style={
                                "width": "100%",
                                "height": "110px",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "boxSizing": "border-box",
                                "marginBottom": "15px"
                            }
                        ),

                        boton(
                            "GENERAR INFORME PDF",
                            "pdf",
                            GREEN
                        ),

                        html.P(
                            "",
                            id="mensaje-pdf",
                            style={"color": YELLOW}
                        )
                    ]),

                    panel([
                        html.H3(
                            "Contenido del informe"
                        ),

                        html.P(
                            "Datos ingresados, fecha y hora, "
                            "gráfica ECG, BPM, RR mediano, "
                            "SDNN, amplitud R, QRS estimado, "
                            "calidad de señal y observaciones."
                        ),

                        html.P(
                            "Sin logo. Informe educativo, "
                            "no diagnóstico.",
                            style={"color": MUTED}
                        )
                    ])
                ]
            )
        ]
    ),

    dcc.Store(
        id="datos",
        data={
            "t": [],
            "y": [],
            "v": [],
            "total": 0,
            "errores": 0,
            "modo": "simulado",
            "calibracion": [1000, 0, 1]
        }
    ),

    dcc.Store(id="activo", data=False),

    dcc.Interval(
        id="temporizador",
        interval=500,
        disabled=True
    ),

    dcc.Download(id="descargar_csv"),
    dcc.Download(id="descargar_pdf"),

    html.P(
        "Uso educativo. No apto para diagnóstico médico.",
        style={
            "textAlign": "center",
            "color": MUTED,
            "marginTop": "25px"
        }
    )

], style={
    "backgroundColor": BG,
    "color": WHITE,
    "minHeight": "100vh",
    "padding": "22px",
    "fontFamily": "Arial, sans-serif"
})


@app.callback(
    Output("datos", "data"),
    Output("activo", "data"),
    Output("temporizador", "disabled"),
    Output("mensaje", "children"),

    Input("iniciar", "n_clicks"),
    Input("detener", "n_clicks"),
    Input("temporizador", "n_intervals"),

    State("datos", "data"),
    State("activo", "data"),
    State("fuente", "value"),
    State("canal", "value"),
    State("ganancia", "value"),
    State("offset", "value"),
    State("escala", "value"),

    prevent_initial_call=True
)
def adquisicion(
    iniciar, detener, tick,
    datos, activo, fuente,
    canal, ganancia, offset, escala
):

    accion = ctx.triggered_id

    if accion == "detener":
        detener_daq()

        return datos, False, True, "Adquisición detenida."

    if accion == "iniciar":
        try:
            ganancia = float(ganancia)
            offset = float(offset)
            escala = float(escala)

            if not np.all(np.isfinite([
                ganancia, offset, escala
            ])):
                raise ValueError(
                    "Calibración inválida."
                )

            if ganancia <= 0 or escala <= 0:
                raise ValueError(
                    "Ganancia y escala deben ser positivas."
                )

            detener_daq()

            if fuente == "daq":
                iniciar_daq(str(canal).strip())

        except Exception as error:
            return (
                datos,
                False,
                True,
                "Error: " + str(error)
            )

        nuevos = {
            "t": [],
            "y": [],
            "v": [],
            "total": 0,
            "errores": 0,
            "modo": fuente,
            "calibracion": [
                ganancia, offset, escala
            ]
        }

        return (
            nuevos,
            True,
            False,
            "Adquisición iniciada."
        )

    if not activo:
        return (
            no_update,
            no_update,
            no_update,
            no_update
        )

    t_anterior = np.asarray(
        datos["t"], dtype=float
    )

    y_anterior = np.asarray(
        datos["y"], dtype=float
    )

    v_anterior = np.asarray(
        datos["v"], dtype=float
    )

    tiempo_inicial = (
        t_anterior[-1] + 1 / FS
        if len(t_anterior)
        else 0.0
    )

    tiempo = (
        tiempo_inicial + np.arange(BLOCK) / FS
    )

    try:
        if datos["modo"] == "daq":
            voltaje = leer_daq()

            g, o, e = datos["calibracion"]

            señal = (
                (voltaje - o) * 1000 * e / g
            )

        else:
            señal = simular_ecg(tiempo)
            voltaje = np.array([])

        if len(señal) != BLOCK:
            raise RuntimeError(
                "Bloque incompleto."
            )

        if not np.all(np.isfinite(señal)):
            raise RuntimeError(
                "Señal inválida."
            )

    except Exception as error:
        detener_daq()
        datos["errores"] += 1

        return (
            datos,
            False,
            True,
            "Error de adquisición: " + str(error)
        )

    maximo = FS * MAX_SECONDS

    datos["t"] = np.concatenate(
        (t_anterior, tiempo)
    )[-maximo:].tolist()

    datos["y"] = np.concatenate(
        (y_anterior, señal)
    )[-maximo:].tolist()

    if datos["modo"] == "daq":
        datos["v"] = np.concatenate(
            (v_anterior, voltaje)
        )[-maximo:].tolist()

    datos["total"] += len(señal)

    return datos, True, False, ""


@app.callback(
    Output("ecg", "figure"),
    Output("original", "figure"),
    Output("filtrada", "figure"),
    Output("fft", "figure"),
    Output("rrgrafica", "figure"),
    Output("bpm", "children"),
    Output("amplitud", "children"),
    Output("qrs", "children"),
    Output("calidad", "children"),
    Output("rr", "children"),
    Output("sdnn", "children"),
    Output("cantidad", "children"),
    Output("bpm2", "children"),
    Output("muestras", "children"),
    Output("perdidas", "children"),
    Output("duracion", "children"),
    Output("conexion", "children"),
    Output("errores", "children"),

    Input("datos", "data"),
    Input("activo", "data"),
    Input("filtro", "value")
)
def actualizar(datos, activo, filtro):
    t = np.asarray(datos["t"], dtype=float)
    y = np.asarray(datos["y"], dtype=float)
    v = np.asarray(datos["v"], dtype=float)

    limite = max(
        0, len(t) - VIEW_SECONDS * FS
    )

    tiempo = t[limite:]
    señal = y[limite:]

    procesada = filtrar(señal, filtro)
    analisis = analizar_ecg(tiempo, señal)

    calidad = calidad_senal(
        señal,
        v[-len(señal):] if len(v) else [],
        datos["errores"]
    )

    g_ecg = grafica(
        tiempo,
        procesada,
        "Monitor ECG",
        analisis["picos"]
    )

    g_original = grafica(
        tiempo, señal, "Original"
    )

    g_filtrada = grafica(
        tiempo, procesada, "Filtrada"
    )

    g_fft = grafica_fft(procesada)
    g_rr = grafica_rr(analisis["rr"])

    bpm = (
        "{:.0f} BPM".format(analisis["bpm"])
        if analisis["bpm"] is not None
        else "--"
    )

    amplitud = (
        "{:.3f} mV".format(analisis["amplitud"])
        if analisis["amplitud"] is not None
        else "--"
    )

    qrs = (
        "{:.0f} ms".format(analisis["qrs"])
        if analisis["qrs"] is not None
        else "--"
    )

    rr = (
        "{:.0f} ms".format(analisis["rr_medio"])
        if analisis["rr_medio"] is not None
        else "--"
    )

    sdnn = (
        "{:.1f} ms".format(analisis["sdnn"])
        if analisis["sdnn"] is not None
        else "--"
    )

    estado = (
        "DAQ ACTIVA"
        if activo and datos["modo"] == "daq"
        else "SIMULACIÓN"
        if activo
        else "DETENIDO"
    )

    return (
        g_ecg,
        g_original,
        g_filtrada,
        g_fft,
        g_rr,
        bpm,
        amplitud,
        qrs,
        calidad,
        rr,
        sdnn,
        str(analisis["cantidad"]),
        bpm,
        str(datos["total"]),
        "NO VERIFICADAS",
        "{:.1f} s".format(datos["total"] / FS),
        estado,
        str(datos["errores"])
    )


def crear_csv(datos):
    archivo = io.StringIO()
    escritor = csv.writer(archivo)

    escritor.writerow([
        "Tiempo_s", "ECG_mV"
    ])

    escritor.writerows(
        zip(datos["t"], datos["y"])
    )

    return archivo.getvalue()


@app.callback(
    Output("descargar_csv", "data"),
    Input("csv", "n_clicks"),
    State("datos", "data"),
    prevent_initial_call=True
)
def descargar_csv(n, datos):
    if not datos["t"]:
        return no_update

    nombre = "ECG_{}.csv".format(
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    return dict(
        content=crear_csv(datos),
        filename=nombre
    )


@app.callback(
    Output("guardado", "children"),
    Input("guardar", "n_clicks"),
    State("datos", "data"),
    prevent_initial_call=True
)
def guardar_sesion(n, datos):
    if not datos["t"]:
        return "No hay datos para guardar."

    carpeta = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "registros"
    )

    os.makedirs(carpeta, exist_ok=True)

    nombre = "ECG_{}.csv".format(
        datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    ruta = os.path.join(carpeta, nombre)

    with open(
        ruta, "w",
        encoding="utf-8",
        newline=""
    ) as archivo:
        archivo.write(crear_csv(datos))

    return "Sesión guardada: " + nombre


def valor_pdf(valor, unidad="", decimales=1):
    if valor is None:
        return "No disponible"

    return (
        ("{:." + str(decimales) + "f} {}")
        .format(valor, unidad)
        .strip()
    )


def generar_pdf(
    datos, paciente, edad, sexo,
    responsable, derivacion, observaciones
):
    t = np.asarray(datos["t"], dtype=float)
    y = np.asarray(datos["y"], dtype=float)
    v = np.asarray(datos["v"], dtype=float)

    if len(t) == 0:
        raise ValueError("No hay datos registrados.")

    limite = max(
        0, len(t) - VIEW_SECONDS * FS
    )

    tw = t[limite:]
    yw = y[limite:]

    analisis = analizar_ecg(tw, yw)

    calidad = calidad_senal(
        yw,
        v[-len(yw):] if len(v) else [],
        datos["errores"]
    )

    imagen = io.BytesIO()

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(
        tw, yw,
        color="#147d91",
        linewidth=1
    )

    picos = np.asarray(
        analisis["picos"],
        dtype=int
    )

    if len(picos):
        ax.scatter(
            tw[picos],
            yw[picos],
            color="#e59a32",
            s=15,
            label="Picos R"
        )
        ax.legend(loc="upper right")

    ax.set_title("Electrocardiograma")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (mV)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        imagen,
        format="png",
        dpi=140
    )

    plt.close(fig)
    imagen.seek(0)

    salida = io.BytesIO()

    pdf = canvas.Canvas(
        salida,
        pagesize=(595, 842)
    )

    pdf.setTitle(
        "ECG LAB - Informe de sesión"
    )

    pdf.setFillColorRGB(0.04, 0.08, 0.14)
    pdf.rect(
        0, 0, 595, 842,
        fill=1, stroke=0
    )

    pdf.setFillColorRGB(0.32, 0.84, 0.92)
    pdf.setFont("Helvetica-Bold", 22)

    pdf.drawString(
        42, 790,
        "ECG LAB - INFORME"
    )

    pdf.setFillColorRGB(0.90, 0.95, 1)
    pdf.setFont("Helvetica", 10)

    pdf.drawString(
        42, 766,
        "Fecha: " + datetime.now().strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    )

    campos = [
        ("Paciente / código", paciente or "No indicado"),
        ("Edad", str(edad) if edad is not None else "No indicada"),
        ("Sexo", sexo or "No indicado"),
        ("Responsable", responsable or "No indicado"),
        ("Derivación", derivacion or "No indicada"),
        ("Fuente", datos["modo"])
    ]

    altura = 738

    for titulo, valor in campos:
        pdf.drawString(
            42, altura,
            (titulo + ": " + str(valor))[:90]
        )
        altura -= 20

    pdf.setFillColorRGB(0.32, 0.84, 0.92)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(
        42, 600,
        "Resultados experimentales"
    )

    resultados = [
        ("Frecuencia cardíaca",
         valor_pdf(analisis["bpm"], "BPM", 0)),
        ("Amplitud R estimada",
         valor_pdf(analisis["amplitud"], "mV", 3)),
        ("QRS estimado",
         valor_pdf(analisis["qrs"], "ms", 0)),
        ("RR mediano",
         valor_pdf(analisis["rr_medio"], "ms", 0)),
        ("SDNN",
         valor_pdf(analisis["sdnn"], "ms", 1)),
        ("Calidad de señal", calidad),
        ("Muestras recibidas", str(datos["total"])),
        ("Errores detectados", str(datos["errores"])),
        ("Duración",
         "{:.1f} s".format(datos["total"] / FS))
    ]

    pdf.setFillColorRGB(0.90, 0.95, 1)
    pdf.setFont("Helvetica", 10)

    altura = 577

    for titulo, valor in resultados:
        pdf.drawString(
            42, altura,
            titulo + ": " + valor
        )
        altura -= 18

    pdf.drawImage(
        ImageReader(imagen),
        42, 225,
        width=510,
        height=165
    )

    pdf.setFillColorRGB(0.32, 0.84, 0.92)
    pdf.setFont("Helvetica-Bold", 12)

    pdf.drawString(
        42, 200,
        "Observaciones"
    )

    pdf.setFillColorRGB(0.90, 0.95, 1)
    pdf.setFont("Helvetica", 9)

    texto = pdf.beginText(42, 182)
    texto.setLeading(13)

    contenido = str(observaciones or "Sin observaciones.")

    import textwrap

    lineas = []

    for parrafo in contenido.splitlines():
        lineas.extend(
            textwrap.wrap(
                parrafo,
                width=95,
                break_long_words=True
            ) or [""]
        )

    for linea in lineas[:6]:
        texto.textLine(
            linea.encode(
                "latin-1", "replace"
            ).decode("latin-1")
        )

    pdf.drawText(texto)

    pdf.setFont("Helvetica", 8)
    pdf.setFillColorRGB(0.9, 0.72, 0.45)

    pdf.drawString(
        42, 45,
        "Uso educativo. Mediciones automáticas no validadas clínicamente."
    )

    pdf.save()

    return salida.getvalue()


@app.callback(
    Output("descargar_pdf", "data"),
    Output("mensaje-pdf", "children"),

    Input("pdf", "n_clicks"),

    State("datos", "data"),
    State("paciente", "value"),
    State("edad", "value"),
    State("sexo", "value"),
    State("responsable", "value"),
    State("derivacion", "value"),
    State("observaciones", "value"),

    prevent_initial_call=True
)
def descargar_pdf(
    n, datos, paciente, edad, sexo,
    responsable, derivacion, observaciones
):
    if not datos["t"]:
        return (
            no_update,
            "Primero debes registrar una señal ECG."
        )

    try:
        contenido = generar_pdf(
            datos,
            paciente,
            edad,
            sexo,
            responsable,
            derivacion,
            observaciones
        )

    except Exception as error:
        return no_update, "Error: " + str(error)

    nombre = "Informe_ECG_{}.pdf".format(
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    return (
        dcc.send_bytes(contenido, nombre),
        "Informe generado correctamente."
    )


if __name__ == "__main__":
    app.run_server(
        host="127.0.0.1",
        port=8050,
        debug=False
    )
