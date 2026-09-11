"""
Central configuration for the action recognition project.
Edit CLASSES to change which UCF101 action categories you train on.
"""

import os

# ---- Class subset -----------------------------------------------------
# Starting with a 10-class sports/movement subset of UCF101 instead of
# the full 101 classes. This keeps training feasible on a single GPU
# and mirrors the kind of clips you already worked with in KOLSA.
# All names must exactly match UCF101's folder naming (CamelCase, no spaces).
CLASSES = [
    "JugglingBalls",
    "JumpRope",
    "JumpingJack",
    "WalkingWithDog",
    "Basketball",
    "TennisSwing",
    "GolfSwing",
    "PushUps",
    "PullUps",
    "Bowling",
]
NUM_CLASSES = len(CLASSES)

# ---- Paths --------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UCF101_subset")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ---- Video sampling -----------------------------------------------------
NUM_FRAMES = 16          # frames sampled per clip
FRAME_SIZE = 112         # H = W = 112, matches r2plus1d_18 pretraining
TRAIN_SPLIT = 0.8        # per-class random split (see data/download_ucf101.py)

# ---- Training -------------------------------------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

DEVICE = "cuda"  # falls back to "cpu" automatically in train.py if unavailable
