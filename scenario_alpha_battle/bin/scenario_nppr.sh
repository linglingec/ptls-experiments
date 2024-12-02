#!/bin/bash

# Script to include and run main.py with an embedded configuration

# Define the embedded Python script (main.py)
MAIN_PY="main.py"

# Extract the embedded Python script
cat << 'EOF' > $MAIN_PY
# Content of main.py starts here
import os
import yaml
import logging
import argparse

import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import get_dataloader, InferenceCallback
from nppr_module import NpprPretrainModule, NpprInferenceModule

# Set up logging
logging.basicConfig(
    filename="progress_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting main.py execution.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the training pipeline.")
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to the configuration file (default: config.yaml)"
)
parser.add_argument(
    "--log_file",
    type=str,
    default="progress_log.txt",
    help="Path to the log file (default: progress_log.txt)"
)
args = parser.parse_args()

# Update logging file based on the argument
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
try:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully from %s.", args.config)
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    raise

# The rest of your main.py code goes here
EOF

# Make sure arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 path_to_config [log_file]"
  exit 1
fi

CONFIG_FILE=$1
LOG_FILE=${2:-"progress_log.txt"} # Default log file is progress_log.txt

# Run the extracted main.py
python3 $MAIN_PY --config "$CONFIG_FILE" --log_file "$LOG_FILE"

# Clean up by removing the extracted Python script
rm -f $MAIN_PY
