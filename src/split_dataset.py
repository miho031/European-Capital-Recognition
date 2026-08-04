from pathlib import Path
import random
import shutil


DATASET_ROOT = Path("dataset/raw")
OUTPUT_ROOT = Path("data")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

cities = [
    folder
    for folder in DATASET_ROOT.iterdir()
    if folder.is_dir()
]
