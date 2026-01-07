# Use Python 3.11.9 as base image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create dummy vertex_credentials.json files if they don't exist
# One in /app for when DIRECTORY=/app, one in agent_arch for local path
RUN if [ ! -f vertex_credentials.json ]; then \
    echo '{}' > vertex_credentials.json; \
    fi && \
    if [ ! -f agent_arch/vertex_credentials.json ]; then \
    echo '{}' > agent_arch/vertex_credentials.json; \
    fi

# Create directories with proper permissions
RUN mkdir -p results agent_arch && \
    chmod -R 777 results agent_arch

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]