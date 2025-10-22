#!/bin/bash

# DDP Launch Script for Full NFSBA Workflow (Generator -> Victim -> Evaluate)

#--- Configuration ---
NNODES=1                 # Number of machines (nodes)
NODE_RANK=0              # Rank of the current machine (0 if only one machine)
NPROC_PER_NODE=2         # Number of GPUs per machine
MASTER_ADDR='localhost'  # Address of the master node (rank 0)
MASTER_PORT='12355'      # Free port for communication (Ensure it's free or change it)
CONFIG_FILE='configs/nfsba_cifar10.yaml' # Your final config file

# --- Script Names ---
GENERATOR_SCRIPT='train_generator.py'
VICTIM_SCRIPT='train_victim.py'
EVALUATE_SCRIPT='evaluate.py'

# --- Determine Absolute Paths ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
GENERATOR_SCRIPT_PATH="$PROJECT_ROOT/$GENERATOR_SCRIPT"
VICTIM_SCRIPT_PATH="$PROJECT_ROOT/$VICTIM_SCRIPT"
EVALUATE_SCRIPT_PATH="$PROJECT_ROOT/$EVALUATE_SCRIPT"
ABS_CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"

# --- Function to check if file exists ---
check_file_exists() {
  if [ ! -f "$1" ]; then
    echo "ERROR: Script or Config not found at '$1'"
    exit 1
  fi
}

# --- Verify paths ---
echo "Project Root: $PROJECT_ROOT"
echo "Config File Path: $ABS_CONFIG_PATH"
check_file_exists "$GENERATOR_SCRIPT_PATH"
check_file_exists "$VICTIM_SCRIPT_PATH"
check_file_exists "$EVALUATE_SCRIPT_PATH"
check_file_exists "$ABS_CONFIG_PATH"
echo "Current Working Directory: $(pwd)" # Should be nfsba/

# ==================================
# STEP 1: Train the Generator
# ==================================
echo ""
echo "--- Starting STEP 1: Generator Training ---"
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$GENERATOR_SCRIPT_PATH" --config "$ABS_CONFIG_PATH"

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "--- ERROR: Generator Training failed with exit code $EXIT_CODE. Aborting. ---"
  exit $EXIT_CODE
else
  echo "--- Generator Training finished successfully. ---"
fi

# ==================================
# STEP 2: Train the Victim Model
# ==================================
echo ""
echo "--- Starting STEP 2: Victim Model Training ---"
# Victim training needs the generator checkpoint from Step 1.
# Ensure train_victim.py and the config file correctly point to the trained generator.
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$VICTIM_SCRIPT_PATH" --config "$ABS_CONFIG_PATH"

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "--- ERROR: Victim Model Training failed with exit code $EXIT_CODE. Aborting. ---"
  exit $EXIT_CODE
else
  echo "--- Victim Model Training finished successfully. ---"
fi

# ==================================
# STEP 3: Evaluate the Final Model
# ==================================
echo ""
echo "--- Starting STEP 3: Final Evaluation ---"
# Evaluation script runs on a single process/GPU (usually rank 0 implicitly handles saving)
# It needs both the final generator and final victim checkpoints.
# Ensure evaluate.py and the config file point to the correct trained models.
python "$EVALUATE_SCRIPT_PATH" --config "$ABS_CONFIG_PATH"

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "--- ERROR: Evaluation failed with exit code $EXIT_CODE. ---"
  exit $EXIT_CODE
else
  echo "--- Evaluation finished successfully. ---"
fi

echo ""
echo "--- Full Workflow Completed ---"

# --- How to Run ---
# 1. Save this script as launch_ddp.sh in the scripts/ directory.
# 2. Make it executable: chmod +x scripts/launch_ddp.sh
# 3. Ensure the CONFIG_FILE variable points to your final, correctly configured YAML file.
# 4. Run it *from the project root directory* (nfsba/): ./scripts/launch_ddp.sh
# 5. Make sure MASTER_PORT is not already in use.