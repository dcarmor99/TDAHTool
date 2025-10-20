FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 👇 Ojo: estamos dentro del contexto ./backend, así que es 'requirements.txt'
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 👇 Copiamos TODO el backend (api.py, utils/, model/, __init__.py…) a /app/backend
COPY . /app/backend

EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
