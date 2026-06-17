FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "uvicorn", "app.fact_checker:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
