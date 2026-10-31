"""
Central Configuration Module for ICAIN 2026 Corel-1K CBIR Implementation.
Single Source of Truth for all dataset, model, training, and evaluation parameters.
"""

from pathlib import Path
import os

# ==========================================
# 1. Dataset Configuration
# ==========================================
LOCAL_DATASET_PATH = Path(__file__).resolve().parent / "dataset"
DEFAULT_DATASET_PATH = LOCAL_DATASET_PATH if LOCAL_DATASET_PATH.exists() else Path(r"C:\Users\777Pr\Downloads\archive (1)\dataset")
DATASET_PATH = Path(os.getenv("COREL_DATASET_PATH", str(DEFAULT_DATASET_PATH))).resolve()

# Expected categories in Corel-1K dataset (10 categories, 100 images each)
EXPECTED_CATEGORIES = [
    "beaches",
    "bus",
    "dinosaurs",
    "elephants",
    "flowers",
    "foods",
    "horses",
    "monuments",
    "mountains_and_snow",
    "peolpe_and_villages_in_Africa"
]

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================
# 2. Experiment & Split Parameters
# ==========================================
RANDOM_SEED = 42
MLP_SEED = 42

# Main benchmark split targets
TOTAL_IMAGES_PER_CAT = 100
GALLERY_IMAGES_PER_CAT = 90
QUERY_IMAGES_PER_CAT = 10

# Internal training/validation split on gallery
INTERNAL_TRAIN_PER_CAT = 70
INTERNAL_VAL_PER_CAT = 20

# Near-duplicate audit threshold (Hamming distance on 64-bit DCT pHash)
NEAR_DUPLICATE_HAMMING_THRESHOLD = 4

# Known near-duplicate pair in Corel-1K
KNOWN_NEAR_DUPLICATE_PAIR = frozenset((
    "training_set/horses/761.jpg",
    "training_set/horses/776.jpg"
))
EXCLUDED_GALLERY_DUPLICATES = [
    "training_set/horses/761.jpg"
]

# Expected split counts for corrected protocol
EXPECTED_COUNTS = {
    "dataset_images": 1000,
    "gallery": 899,
    "queries": 100,
    "excluded_duplicate_gallery": 1,
    "mlp_train": 700,
    "mlp_validation": 199
}

# ==========================================
# 3. Feature Extraction Parameters
# ==========================================
# HSV Color Moments (Mean, Std, Skewness for H, S, V) -> 9 dimensions
COLOR_FEATURE_DIM = 9

# GLCM Texture Parameters (Contrast, Correlation, Energy, Homogeneity) -> 4 dimensions
TEXTURE_FEATURE_DIM = 4
GLCM_DISTANCE = 1
GLCM_ANGLES_DEG = [0, 45, 90, 135]

# Total Feature Dimension
TOTAL_FEATURE_DIM = COLOR_FEATURE_DIM + TEXTURE_FEATURE_DIM  # 13 dimensions

# ==========================================
# 4. Baseline Model Fusion Weights
# ==========================================
# M3: Equal Feature-Level Fusion (wc=0.5, wt=0.5)
M3_COLOR_WEIGHT = 0.5
M3_TEXTURE_WEIGHT = 0.5

# M4: Fixed Feature-Level Fusion (wc=0.7, wt=0.3)
M4_COLOR_WEIGHT = 0.7
M4_TEXTURE_WEIGHT = 0.3

# ==========================================
# 5. MLP Architecture & Training Parameters
# ==========================================
MLP_INPUT_DIM = TOTAL_FEATURE_DIM  # 13
MLP_HIDDEN_1 = 16
MLP_HIDDEN_2 = 8
MLP_OUTPUT_DIM = 2
MLP_LEAKY_RELU_SLOPE = 0.1

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
MARGIN = 0.15  # Triplet Margin Loss
EARLY_STOPPING_PATIENCE = 10

# Triplet sampling parameters
TRAIN_POS_SAMPLES = 5
TRAIN_NEG_SAMPLES = 10
VAL_POS_SAMPLES = 2
VAL_NEG_SAMPLES = 5

# Reference audit checkpoint epoch (for verification)
REFERENCE_AUDIT_CHECKPOINT_EPOCH = 8

# ==========================================
# 6. Evaluation Parameters
# ==========================================
EVAL_K = 10

# Reference M6 Fixed Distance-Level Control Weights (derived from reference run query predictions)
M6_REFERENCE_COLOR_WEIGHT = 0.5784038992226124
M6_REFERENCE_TEXTURE_WEIGHT = 0.42159609973430634

# ==========================================
# 7. Output Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PHASE1_OUTPUT_DIR = OUTPUT_DIR / "phase1"
PHASE2_OUTPUT_DIR = OUTPUT_DIR / "phase2"
PHASE3_OUTPUT_DIR = OUTPUT_DIR / "phase3"
PHASE4_OUTPUT_DIR = OUTPUT_DIR / "phase4"
PHASE5_OUTPUT_DIR = OUTPUT_DIR / "phase5"
PHASE6_OUTPUT_DIR = OUTPUT_DIR / "phase6"

FIGURES_DIR = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "model"
MODEL_CHECKPOINT_PATH = MODEL_DIR / "adaptive_weight_mlp.pt"
PART2_SEED = 42
