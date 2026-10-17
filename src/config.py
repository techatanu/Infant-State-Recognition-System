"""Central configuration. Values mirror CONFIG in Baby_Cry_Scene.ipynb."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Dataset lives outside the repo; set BABYCRY_DATA_DIR in .env (see .env.example)
DATASET_PATH = Path(os.environ.get("BABYCRY_DATA_DIR", ROOT / "data" / "raw"))

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CACHE_DIR = ROOT / "data" / "cache"

CLASSES = ["belly_pain", "burping", "cold_hot", "discomfort",
           "hungry", "lonely", "scared", "tired"]

# Audio / feature params
SAMPLE_RATE = 22050
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = int(SAMPLE_RATE * 0.010)   # 10 ms -> 220
WIN_LENGTH = int(SAMPLE_RATE * 0.025)   # 25 ms -> 551
N_MELS_RF = 40
N_FEATURES = 218

# CNN log-mel input
N_MELS_CNN = 64
CNN_N_FFT = 1024
CNN_HOP = 256
CNN_DURATION_SEC = 7.0                      # ~90% of trimmed clips are <= 7 s
CNN_N_SAMPLES = int(SAMPLE_RATE * CNN_DURATION_SEC)
CNN_N_FRAMES = 1 + CNN_N_SAMPLES // CNN_HOP  # 603

# Cleaning rules
TRIM_TOP_DB = 30
MIN_TRIMMED_SEC = 1.0
MAX_SILENCE_PCT = 80

# Split / model
TEST_SIZE = 0.2
SEED = 42
N_FOLDS = 5
