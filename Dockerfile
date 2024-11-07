# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias del sistema necesarias para FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de requisitos
COPY backend/requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Crea los directorios necesarios
RUN mkdir -p backend/app/faiss_index \
    && mkdir -p backend/app/processed_scripts \
    && mkdir -p backend/app/scripts

# Copia el código de la aplicación
COPY backend/app backend/app
COPY backend/embeddings.pkl backend/embeddings.pkl

# Expone el puerto
EXPOSE 8000

# Establece las variables de entorno
ENV PYTHONPATH=/app

# Comando para ejecutar la aplicación
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]