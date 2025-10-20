#!/bin/bash

# Simple DDP Launch Script Example for 2 GPUs

#--- Configuration ---
NNODES=1                 # Number of machines (nodes)
NODE_RANK=0              # Rank of the current machine (0 if only one machine)
NPROC_PER_NODE=2         # Number of GPUs per machine (e.g., 2 for two 4090s)
MASTER_ADDR='localhost'  # Address of the master node (rank 0)
MASTER_PORT='12355'      # Free port for communication (Ensure it's free or change it)
CONFIG_FILE='configs/nfsba_cifar10.yaml' # Your config file

# --- SELECT THE SCRIPT TO RUN ---
# Uncomment the script you want to run:

# ===> STEP 1: Train the generator first <===
TRAIN_SCRIPT_NAME='train_generator.py'

# ===> STEP 2: After generator training completes, comment the line above and uncomment the line below <===
# TRAIN_SCRIPT_NAME='train_victim.py'

# --- DDP Execution ---
echo "Starting DDP training for $TRAIN_SCRIPT_NAME with $NPROC_PER_NODE GPUs..."

# --- Determine Absolute Paths ---
# Get the directory where this launch script itself resides
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Assume the project root is one level above the 'scripts' directory
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# Construct the absolute path to the training script
TRAIN_SCRIPT_PATH="$PROJECT_ROOT/$TRAIN_SCRIPT_NAME"
# Construct the absolute path to the config file (safer)
ABS_CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"

# --- Verify paths (optional debugging) ---
echo "Project Root: $PROJECT_ROOT"
echo "Training Script Path: $TRAIN_SCRIPT_PATH"
echo "Config File Path: $ABS_CONFIG_PATH"
echo "Current Working Directory: $(pwd)" # Should be nfsba/ when you run ./scripts/launch_ddp.sh

# --- Check if script exists ---
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found at '$TRAIN_SCRIPT_PATH'"
    exit 1
fi
if [ ! -f "$ABS_CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at '$ABS_CONFIG_PATH'"
    exit 1
fi


# --- Use the ABSOLUTE PATH for the training script ---
# Note: Using python -m torch.distributed.run which is the recommended launcher
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$TRAIN_SCRIPT_PATH" --config "$ABS_CONFIG_PATH" \
    # Add any other script-specific arguments here if needed

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "DDP training finished successfully."
else
  echo "DDP training failed with exit code $EXIT_CODE."
fi


# --- How to Run ---
# 1. Save this script as launch_ddp.sh in the scripts/ directory.
# 2. Make it executable: chmod +x scripts/launch_ddp.sh
# 3. Edit the TRAIN_SCRIPT_NAME variable inside the script to select
#    'train_generator.py' (first run) or 'train_victim.py' (second run).
# 4. Run it *from the project root directory* (nfsba/): ./scripts/launch_ddp.sh
# 5. Make sure MASTER_PORT is not already in use.