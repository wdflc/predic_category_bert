import csv
import json
import os
from pathlib import Path


# Repository layout used by the existing code:
# data/{train.txt,test.txt,valid.txt}
# data/processed/{train,test,valid}
# pretrained/bert-base-chinese
# models, logs
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"
PRE_TRAINED_DIR = BASE_DIR / "pretrained"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
BERT_MODEL_DIR = str(PRE_TRAINED_DIR / "bert-base-chinese")

SEQ_LEN = int(os.getenv("SEQ_LEN", "64"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
PATIENCE = int(os.getenv("PATIENCE", "2"))


def _num_classes_from_processed_train() -> int | None:
    dataset_info = PROCESSED_DATA_DIR / "train" / "dataset_info.json"
    if not dataset_info.exists():
        return None

    try:
        with dataset_info.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    label_feature = info.get("features", {}).get("label", {})
    names = label_feature.get("names")
    if isinstance(names, list) and names:
        return len(names)
    return None


def _num_classes_from_raw_train() -> int | None:
    train_file = RAW_DATA_DIR / "train.txt"
    if not train_file.exists():
        return None

    labels = set()
    try:
        with train_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if "label" not in (reader.fieldnames or []):
                return None
            for row in reader:
                label = row.get("label")
                if label:
                    labels.add(label)
    except OSError:
        return None

    return len(labels) if labels else None


def _detect_num_classes() -> int:
    env_value = os.getenv("NUM_CLASSES")
    if env_value:
        return int(env_value)

    detected = _num_classes_from_processed_train()
    if detected is not None:
        return detected

    detected = _num_classes_from_raw_train()
    if detected is not None:
        return detected

    return 2


def _detect_device() -> str:
    env_value = os.getenv("DEVICE")
    if env_value:
        return env_value

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


NUM_CLASSES = _detect_num_classes()
DEVICE = _detect_device()

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PRE_TRAINED_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
