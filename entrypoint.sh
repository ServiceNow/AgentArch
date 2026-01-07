#!/bin/bash
set -e

# Check if MODEL_TO_RUN is set
if [ -z "$MODEL_TO_RUN" ]; then
    echo "Error: MODEL_TO_RUN environment variable is not set"
    exit 1
fi

echo "================================================"
echo "Running benchmark with model: $MODEL_TO_RUN"
echo "================================================"
echo ""

# Use cases
USECASES=("requesting_time_off" "customer_request_routing")

# Modes
MODES=("single_agent" "direct" "indirect")

# Agent types
AGENT_TYPES=("function_calling" "ReAct")

# Memory types
MEMORY_TYPES=("transparent" "compact")

# Default batch size
BATCH_SIZE="${BATCH_SIZE:-70}"

# Default project name
PROJECT="${PROJECT:-default}"

# Debug mode
DEBUG_FLAG=""
if [ "${DEBUG}" = "true" ] || [ "${DEBUG}" = "True" ] || [ "${DEBUG}" = "1" ]; then
    DEBUG_FLAG="--debug"
    echo "🐛 Debug mode enabled"
fi

# PASS_K flag
PASS_K_FLAG=""
if [ -n "$K" ]; then
    PASS_K_FLAG="--pass_k $K"
    echo "📊 Pass K set to: $K"
fi

# Parse SKIP_CONFIGS into an array
declare -A SKIP_CONFIG_MAP
if [ -n "$SKIP_CONFIGS" ]; then
    echo "📋 Parsing skip configs: $SKIP_CONFIGS"
    IFS=',' read -ra SKIP_ARRAY <<< "$SKIP_CONFIGS"
    for config_num in "${SKIP_ARRAY[@]}"; do
        # Trim whitespace
        config_num=$(echo "$config_num" | xargs)
        SKIP_CONFIG_MAP[$config_num]=1
    done
    echo "⏭️  Will skip configurations: ${!SKIP_CONFIG_MAP[@]}"
    echo ""
fi

TOTAL_CONFIGS=0
SUCCESSFUL_CONFIGS=0
FAILED_CONFIGS=0
SKIPPED_CONFIGS=0

# Function to run a configuration
run_config() {
    local usecase=$1
    local mode=$2
    local agent_type=$3
    local memory_type=$4
    local thinking=$5
    local config_num=$6

    # Check if this config should be skipped
    if [ -n "${SKIP_CONFIG_MAP[$config_num]}" ]; then
        echo ""
        echo "----------------------------------------"
        echo "Config $config_num: SKIPPED"
        echo "  Usecase: $usecase"
        echo "  Mode: $mode"
        echo "  Agent Type: $agent_type"
        echo "  Memory: $memory_type"
        echo "  Thinking: $thinking"
        echo "----------------------------------------"
        SKIPPED_CONFIGS=$((SKIPPED_CONFIGS + 1))
        return
    fi

    echo ""
    echo "----------------------------------------"
    echo "Config $config_num:"
    echo "  Usecase: $usecase"
    echo "  Mode: $mode"
    echo "  Agent Type: $agent_type"
    echo "  Memory: $memory_type"
    echo "  Thinking: $thinking"
    echo "----------------------------------------"

    local thinking_flag=""
    if [ "$thinking" == "enabled" ]; then
        thinking_flag="--thinking-tools-enabled"
    fi

    python agent_arch/run.py \
        --model "$MODEL_TO_RUN" \
        --usecase "$usecase" \
        --mode "$mode" \
        --agent_type "$agent_type" \
        --memory_management "$memory_type" \
        $thinking_flag \
        --project "$PROJECT" \
        --directory /app/agent_arch \
        $DEBUG_FLAG \
        $PASS_K_FLAG

    if [ $? -eq 0 ]; then
        echo "✅ Configuration completed successfully"
        SUCCESSFUL_CONFIGS=$((SUCCESSFUL_CONFIGS + 1))
    else
        echo "❌ Configuration failed"
        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
    fi
}

CONFIG_NUM=0

# First pass: Run all configurations WITHOUT thinking tools
for usecase in "${USECASES[@]}"; do
    echo ""
    echo "================================================"
    echo "Starting use case: $usecase (without thinking tools)"
    echo "================================================"

    for mode in "${MODES[@]}"; do
        for agent_type in "${AGENT_TYPES[@]}"; do
            for memory_type in "${MEMORY_TYPES[@]}"; do
                # Skip ReAct with compact memory (not supported)
                if [ "$agent_type" == "ReAct" ] && [ "$memory_type" == "compact" ]; then
                    continue
                fi

                CONFIG_NUM=$((CONFIG_NUM + 1))
                TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
                run_config "$usecase" "$mode" "$agent_type" "$memory_type" "disabled" "$CONFIG_NUM"
            done
        done
    done

    echo ""
    echo "================================================"
    echo "Completed use case: $usecase (without thinking tools)"
    echo "================================================"
done

# Second pass: Run all configurations WITH thinking tools
for usecase in "${USECASES[@]}"; do
    echo ""
    echo "================================================"
    echo "Starting use case: $usecase (with thinking tools)"
    echo "================================================"

    for mode in "${MODES[@]}"; do
        for agent_type in "${AGENT_TYPES[@]}"; do
            for memory_type in "${MEMORY_TYPES[@]}"; do
                # Skip ReAct with compact memory (not supported)
                if [ "$agent_type" == "ReAct" ] && [ "$memory_type" == "compact" ]; then
                    continue
                fi

                CONFIG_NUM=$((CONFIG_NUM + 1))
                TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
                run_config "$usecase" "$mode" "$agent_type" "$memory_type" "enabled" "$CONFIG_NUM"
            done
        done
    done

    echo ""
    echo "================================================"
    echo "Completed use case: $usecase (with thinking tools)"
    echo "================================================"
done

echo ""
echo "================================================"
echo "BENCHMARK COMPLETE!"
echo "================================================"
echo "Total configurations: $TOTAL_CONFIGS"
echo "Successful: $SUCCESSFUL_CONFIGS"
echo "Failed: $FAILED_CONFIGS"
echo "Skipped: $SKIPPED_CONFIGS"
echo "Results saved to: /app/results"
echo "================================================"