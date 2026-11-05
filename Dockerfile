# Uma imagem só para o pipeline de treino, a previsão e o app.
# Ela leva apenas código e bibliotecas. Dados e modelo entram pelos volumes do docker-compose.yml,
# então a imagem não carrega nada da EVCOMX. O .dockerignore garante isso.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONPATH=/app

WORKDIR /app
COPY pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY . /app

EXPOSE 8501
CMD ["python", "pipeline/treinar.py", "--listar"]
