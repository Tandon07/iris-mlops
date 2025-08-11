# Use official Python image
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Default command overridden by docker-compose
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

