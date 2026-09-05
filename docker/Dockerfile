FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "src/serving/server.py"]
