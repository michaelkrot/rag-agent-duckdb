# Use slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory and make it a volume
RUN mkdir -p /app/data
VOLUME /app/data

# Run the agent with a default query for testing
CMD ["python", "agent.py", "This is a sample query"]