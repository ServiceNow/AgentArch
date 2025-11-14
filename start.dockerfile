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

# Create entrypoint script that runs all configurations
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if MODEL_TO_RUN is set\n\
if [ -z "$MODEL_TO_RUN" ]; then\n\
    echo "Error: MODEL_TO_RUN environment variable is not set"\n\
    exit 1\n\
fi\n\
\n\
echo "================================================"\n\
echo "Running benchmark with model: $MODEL_TO_RUN"\n\
echo "================================================"\n\
echo ""\n\
\n\
# Use cases\n\
USECASES=("requesting_time_off" "triage_cases")\n\
\n\
# Modes (3 options)\n\
MODES=("direct" "indirect" "single_agent")\n\
\n\
# Agent types (2 options)\n\
AGENT_TYPES=("function_calling" "ReAct")\n\
\n\
# Memory types (2 options: transparent, compact)\n\
MEMORY_TYPES=("transparent" "compact")\n\
\n\
# Set default batch size if not provided\n\
BATCH_SIZE="${BATCH_SIZE:-70}"\n\
\n\
# Set default project name if not provided\n\
PROJECT="${PROJECT:-default}"\n\
\n\
# Check if DEBUG is enabled\n\
DEBUG_FLAG=""\n\
if [ "${DEBUG}" = "true" ] || [ "${DEBUG}" = "True" ] || [ "${DEBUG}" = "1" ]; then\n\
    DEBUG_FLAG="--debug"\n\
    echo "🐛 Debug mode enabled"\n\
fi\n\
\n\
TOTAL_CONFIGS=0\n\
SUCCESSFUL_CONFIGS=0\n\
FAILED_CONFIGS=0\n\
\n\
# Run all configurations\n\
for usecase in "${USECASES[@]}"; do\n\
    echo ""\n\
    echo "================================================"\n\
    echo "Starting use case: $usecase"\n\
    echo "================================================"\n\
    \n\
    for mode in "${MODES[@]}"; do\n\
        for agent_type in "${AGENT_TYPES[@]}"; do\n\
            for memory_type in "${MEMORY_TYPES[@]}"; do\n\
                # Skip ReAct with compact memory (not supported)\n\
                if [ "$agent_type" == "ReAct" ] && [ "$memory_type" == "compact" ]; then\n\
                    continue\n\
                fi\n\
                \n\
                # Run without thinking tools\n\
                TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))\n\
                echo ""\n\
                echo "----------------------------------------"\n\
                echo "Config $TOTAL_CONFIGS:"\n\
                echo "  Usecase: $usecase"\n\
                echo "  Mode: $mode"\n\
                echo "  Agent Type: $agent_type"\n\
                echo "  Memory: $memory_type"\n\
                echo "  Thinking: disabled"\n\
                echo "----------------------------------------"\n\
                \n\
                python agent_arch/run.py \\\n\
                    --model "$MODEL_TO_RUN" \\\n\
                    --usecase "$usecase" \\\n\
                    --mode "$mode" \\\n\
                    --agent_type "$agent_type" \\\n\
                    --memory_management "$memory_type" \\\n\
                    --project "$PROJECT" \\\n\
                    --directory /app/agent_arch \\\n\
                    $DEBUG_FLAG\n\
                \n\
                if [ $? -eq 0 ]; then\n\
                    echo "✅ Configuration completed successfully"\n\
                    SUCCESSFUL_CONFIGS=$((SUCCESSFUL_CONFIGS + 1))\n\
                else\n\
                    echo "❌ Configuration failed"\n\
                    FAILED_CONFIGS=$((FAILED_CONFIGS + 1))\n\
                fi\n\
                \n\
                # Run with thinking tools enabled\n\
                TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))\n\
                echo ""\n\
                echo "----------------------------------------"\n\
                echo "Config $TOTAL_CONFIGS:"\n\
                echo "  Usecase: $usecase"\n\
                echo "  Mode: $mode"\n\
                echo "  Agent Type: $agent_type"\n\
                echo "  Memory: $memory_type"\n\
                echo "  Thinking: enabled"\n\
                echo "----------------------------------------"\n\
                \n\
                python agent_arch/run.py \\\n\
                    --model "$MODEL_TO_RUN" \\\n\
                    --usecase "$usecase" \\\n\
                    --mode "$mode" \\\n\
                    --agent_type "$agent_type" \\\n\
                    --memory_management "$memory_type" \\\n\
                    --thinking-tools-enabled \\\n\
                    --project "$PROJECT" \\\n\
                    --directory /app/agent_arch \\\n\
                    $DEBUG_FLAG\n\
                \n\
                if [ $? -eq 0 ]; then\n\
                    echo "✅ Configuration completed successfully"\n\
                    SUCCESSFUL_CONFIGS=$((SUCCESSFUL_CONFIGS + 1))\n\
                else\n\
                    echo "❌ Configuration failed"\n\
                    FAILED_CONFIGS=$((FAILED_CONFIGS + 1))\n\
                fi\n\
            done\n\
        done\n\
    done\n\
    \n\
    echo ""\n\
    echo "================================================"\n\
    echo "Completed use case: $usecase"\n\
    echo "================================================"\n\
done\n\
\n\
echo ""\n\
echo "================================================"\n\
echo "BENCHMARK COMPLETE!"\n\
echo "================================================"\n\
echo "Total configurations run: $TOTAL_CONFIGS"\n\
echo "Successful: $SUCCESSFUL_CONFIGS"\n\
echo "Failed: $FAILED_CONFIGS"\n\
echo "Results saved to: /app/results"\n\
echo "================================================"\n\
' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]