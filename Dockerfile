# Use slim image for smaller size
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data
VOLUME /app/data

# Default to interactive REPL – great for demos
CMD ["python", "agent.py", "repl"]