# ECG

instrucciones Mac
todo colocar en la terminal
1. revisar version python:
  python3 --version
  se debe usar Python 3.9.6
2. crear entorno virtual (una vez)
   python3 -m venv venv
3. Activar el entorno virtual
   source venv/bin/activate
4. descargar las librerías:
   python -m pip install "dash==2.18.2" "plotly==6.0.1" "numpy==1.26.4" "scipy==1.13.1" "matplotlib==3.7.5" "reportlab==4.2.5"
5. correr el programa
   python ECGWE.py
6. Cerrar:
   lsof -ti:8050 | xargs kill
