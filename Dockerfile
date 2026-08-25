FROM python:3.10-slim

WORKDIR /app

# Install OS deps (if any) and pip
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files and model
COPY api.py .
COPY weather_model.pkl .

EXPOSE 80

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
