from pathlib import Path
import os

# Project Root
PROJECT_ROOT = Path("/cluster/dataset/midatams/inf-extr")

# Data Path
DATA_PATH = PROJECT_ROOT / "data"
DATA_PATH_SEANTIS = DATA_PATH / "raw/seantis/imported_20210507"
DATA_PATH_RSD = DATA_PATH / "raw/reports_with_struct_data/imported_20210507"
DATA_PATH_LABELLED = DATA_PATH / "raw/labelling"

# Preprocessed Data Path
DATA_PATH_PREPROCESSED = DATA_PATH / "preprocessed"

# Results Path
RESULTS_PATH = PROJECT_ROOT / "results"

# Model Path
MODEL_PATH = PROJECT_ROOT / "resources/models"

# METRICS
METRICS_PATH = PROJECT_ROOT / "resources/metrics"

# Thesis path
THESIS_PATH = Path("/cluster/home/eglimar/inf-extr/thesis")