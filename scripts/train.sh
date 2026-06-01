#!/bin/bash

echo "=========================================="
echo "   SPIDER_3 TRAIN SCRIPT"
echo "=========================================="
echo ""

TASK="spider_3"

# Ask for number of environments
read -p "Number of environments [default: 200]: " NUM_ENVS
NUM_ENVS=${NUM_ENVS:-200}
echo ""

# Ask for max iterations
read -p "Max iterations [default: 1000]: " MAX_ITERS
MAX_ITERS=${MAX_ITERS:-1000}
echo ""

# Ask for headless mode
echo "Run mode:"
echo "  1) Headless (no visualization)"
echo "  2) GUI (with visualization)"
echo ""
read -p "Enter choice [default: 1]: " RUN_MODE

case "$RUN_MODE" in
    2) 
        HEADLESS_FLAG=""
        echo "Running with GUI"
        ;;
    *) 
        HEADLESS_FLAG="--headless"
        echo "Running headless"
        ;;
esac

echo ""
echo "=========================================="
echo "  RUNNING TRAIN COMMAND"
echo "=========================================="
echo ""

# Build command
CMD="python scripts/rsl_rl/train.py --task $TASK --num_envs $NUM_ENVS --max_iterations $MAX_ITERS $HEADLESS_FLAG"

echo "Command: $CMD"
echo ""

# Run the command
eval $CMD