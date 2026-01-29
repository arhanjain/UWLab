#!/bin/bash
set -e

echo "=== Starting SageMaker Multi-GPU Training ==="

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Check if config file is provided
if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: No config file provided. Please set CONFIG_PATH environment variable."
    exit 1
fi

echo "Using config file: $CONFIG_PATH"

# Check if distributed training is requested (multi-GPU)
if [ "$NUM_GPUS" -gt 1 ] && [ ! -z "$MY_SCRIPT" ]; then
    echo "Launching with torchrun for $NUM_GPUS GPUs"

    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS $MY_SCRIPT --distributed --config $CONFIG_PATH"

    echo "Executing: $CMD"
    exec $CMD
else
    echo "Running single-GPU mode"
    exec python $MY_SCRIPT --config $CONFIG_PATH
fi