"""Global configuration for the AnimalCLEF26 pipeline."""
import os
from pathlib import Path

# Repository root, resolved from this file so the code runs wherever it is
# cloned. Set ANIMALCLEF_ROOT to keep data/ and models/ outside the repo.
ROOT = Path(os.environ.get("ANIMALCLEF_ROOT",
                           Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "animalclef2026"
IMAGES = DATA / "images"
CACHE = ROOT / "cache"
SUBMISSIONS = ROOT / "submissions"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

for _d in (CACHE, SUBMISSIONS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

SPECIES = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# Species that have labelled training data (database).
SPECIES_WITH_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]

# Short keys used in cache filenames.
SPECIES_KEY = {
    "LynxID2025": "lynx",
    "SalamanderID2025": "salamander",
    "SeaTurtleID2022": "turtle",
    "TexasHornedLizards": "lizard",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Backbone input sizes (matching each model's pretraining resolution).
MODEL_SPECS = {
    "miewid": {"dir": MODELS / "miewid-msv3", "size": 440},
    "mega": {"dir": MODELS / "MegaDescriptor-L-384", "size": 384},
}

SEED = 42


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
