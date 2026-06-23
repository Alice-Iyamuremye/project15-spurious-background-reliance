"""Central configuration for the ERM baseline.

All paths and hyperparameters live here so the rest of the code stays clean.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# On Kaggle: override via the WILDS_DATA_DIR environment variable, or
# by editing DATA_ROOT below to point at your downloaded dataset.
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("WILDS_DATA_DIR", "/kaggle/working/wilds_data"))
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_ROOT / "models"
RESULT_DIR = ARTIFACT_ROOT / "results"
SALiency_DIR = ARTIFACT_ROOT / "saliency"

for d in (DATA_ROOT, MODEL_DIR, RESULT_DIR, SALiency_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Training ------------------------------------------------------------
SEED = 42
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# --- Model ---------------------------------------------------------------
# ResNet-18 is the WILDS-canonical small backbone; fast enough to train
# several variants for ERM-vs-mitigation comparison.
MODEL_NAME = "resnet18"
PRETRAINED = True

# --- Subgroup labels -----------------------------------------------------
# Waterbirds metadata convention (from WILDS):
#   metadata[:, 0] = background   (0 = land, 1 = water)
#   metadata[:, 1] = bird type    (0 = landbird, 1 = waterbird)
# So the 4 groups are indexed 0..3 below:
GROUP_NAMES = [
    "landbird-on-land",     # group 0  -- majority
    "landbird-on-water",    # group 1  -- minority (HARD)
    "waterbird-on-land",    # group 2  -- minority (HARD)
    "waterbird-on-water",   # group 3  -- majority
]
NUM_GROUPS = len(GROUP_NAMES)
