"""
Central configuration for the action recognition project.

Two modes:
  1. Fixed subset (default) — edit CLASSES below.
  2. Auto-discovery — set USE_ALL_CLASSES = True and data/download_ucf101.py
     will use every class folder found in the downloaded UCF101 dataset,
     writing the resulting list to data/UCF101_subset/classes.txt so
     train.py / evaluate.py / visualize_predictions.py all agree on it
     without you editing anything else.
"""

import os

# ---- Class selection ----------------------------------------------------
USE_ALL_CLASSES = False   # set True to train on every class UCF101 has

# Used only when USE_ALL_CLASSES is False. All names must exactly match
# UCF101's folder naming (CamelCase, no spaces).
CLASSES = [
    "JugglingBalls",
    "JumpRope",
    "JumpingJack",
    "WalkingWithDog",
    "Basketball",
    "BasketballDunk",
    "TennisSwing",
    "GolfSwing",
    "PushUps",
    "PullUps",
    "Bowling",
    "BoxingPunchingBag",
    "Skiing",
    "SkyDiving",
    "HorseRiding",
    "SoccerPenalty",
    "SoccerJuggling",
    "RockClimbingIndoor",
    "Archery",
    "Fencing",
    "Diving",
    "CliffDiving",
    "Surfing",
    "TableTennisShot",
    "VolleyballSpiking",
    "HighJump",
    "LongJump",
    "PlayingGuitar",
]

# ---- Paths --------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UCF101_subset")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.txt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def get_classes() -> list:
    """
    Returns the class list actually used to build the dataset on disk.
    If data/download_ucf101.py has already run, classes.txt reflects the
    real discovered/used classes — that always takes priority over the
    hardcoded CLASSES above, so train/evaluate/visualize never disagree
    with what's actually on disk.
    """
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE) as f:
            return [line.strip() for line in f if line.strip()]
    return CLASSES


NUM_CLASSES = len(get_classes())

# ---- Video sampling -----------------------------------------------------
NUM_FRAMES = 16          # frames sampled per clip
FRAME_SIZE = 112         # H = W = 112, matches r2plus1d_18 pretraining
TRAIN_SPLIT = 0.8        # per-group random split (see data/download_ucf101.py)

# ---- Training -------------------------------------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

DEVICE = "cuda"  # falls back to "cpu" automatically in train.py if unavailable

