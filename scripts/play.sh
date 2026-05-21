#!/bin/bash

echo "=========================================="
echo "   SPIDER_3 PLAY SCRIPT"
echo "=========================================="
echo ""

# Default values
DEFAULT_CHECKPOINT_DIR="logs/rsl_rl/spider_velocity_control"
TASK="spider_3"

# Get latest checkpoint directory
LATEST_DIR=$(ls -td $DEFAULT_CHECKPOINT_DIR/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "ERROR: No checkpoint directories found in $DEFAULT_CHECKPOINT_DIR"
    exit 1
fi

echo "Latest checkpoint directory: $LATEST_DIR"
echo ""

# Ask for checkpoint number
echo "Available checkpoints in $LATEST_DIR:"
ls -1 "$LATEST_DIR" | grep "model_" | sed 's/model_//;s/.pt//' | sort -n | tail -5
echo ""

read -p "Enter checkpoint number (or press Enter for latest): " CHECKPOINT_NUM

if [ -z "$CHECKPOINT_NUM" ]; then
    # Get latest checkpoint
    CHECKPOINT=$(ls -1 "$LATEST_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found"
        exit 1
    fi
    CHECKPOINT_NUM=$(basename "$CHECKPOINT" | sed 's/model_//;s/.pt//')
else
    CHECKPOINT="$LATEST_DIR/model_$CHECKPOINT_NUM.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint $CHECKPOINT not found"
        exit 1
    fi
fi

echo "Selected checkpoint: model_$CHECKPOINT_NUM.pt"
echo ""

# Ask for number of environments
read -p "Number of environments [default: 1]: " NUM_ENVS
NUM_ENVS=${NUM_ENVS:-1}
echo ""

# Ask for fixed velocity
echo "Velocity commands: X Y YAW (values between -1 and 1)"
echo "X: forward/backward velocity"
echo "Y: left/right velocity"  
echo "YAW: angular velocity (turning)"
echo ""

read -p "Enter fixed velocity (X Y YAW) or press Enter for random: " VEL_X VEL_Y VEL_YAW

if [ -z "$VEL_X" ] || [ -z "$VEL_Y" ] || [ -z "$VEL_YAW" ]; then
    FIXED_VEL_FLAG=""
    echo "Using random velocity commands"
else
    FIXED_VEL_FLAG="--fixed_velocity $VEL_X $VEL_Y $VEL_YAW"
    echo "Fixed velocity: X=$VEL_X, Y=$VEL_Y, YAW=$VEL_YAW"
fi

echo ""
echo "=========================================="
echo "  RUNNING PLAY COMMAND"
echo "=========================================="
echo ""

# Build command
CMD="python scripts/rsl_rl/play.py --task $TASK --num_envs $NUM_ENVS --checkpoint $CHECKPOINT"

if [ -n "$FIXED_VEL_FLAG" ]; then
    CMD="$CMD $FIXED_VEL_FLAG"
fi

echo "Command: $CMD"
echo ""

# Run the command
eval $CMD