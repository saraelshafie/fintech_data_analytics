FROM python:3.11-slim

# Install PostgreSQL development packages
RUN apt-get update && \
    apt-get install -y libpq-dev bash && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run the main script
CMD ["python", "src/main.py"]
