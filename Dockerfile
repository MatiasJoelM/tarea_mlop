# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY piplines/Model_creator_pkl.py /app/
COPY piplines/api.py /app/
COPY requirements.txt /app/
COPY data/diabetes.csv /input/

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Ejecuta el script para entrenar el modelo
RUN python Model_creator_pkl.py

# Comando para levantar la API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]