"""
Downloads UCF101 (via the Hugging Face mirror) and keeps only the class
subset defined in src/config.py, then creates a train/test split.

Usage:
    python data/download_ucf101.py

Result:
    data/UCF101_subset/train/<ClassName>/*.avi
    data/UCF101_subset/test/<ClassName>/*.avi
"""

import os
import random
import re
import shutil
import sys
import zipfile

from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASSES, DATA_DIR, TRAIN_SPLIT, SEED  # noqa: E402

HF_REPO = "quchenyuan/UCF101-ZIP"
RAW_DIR = os.path.join(os.path.dirname(__file__), "_raw_ucf101")


def download_full_dataset() -> str:
    """Downloads the full UCF101 zip mirror from Hugging Face (one-time, ~7GB)."""
    print(f"Downloading {HF_REPO} from Hugging Face (this can take a while)...")
    local_dir = snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=RAW_DIR)
    return local_dir


def extract_if_needed(local_dir: str) -> str:
    """Extracts the dataset zip(s) if not already extracted."""
    extracted_marker = os.path.join(local_dir, "_extracted")
    if os.path.exists(extracted_marker):
        return local_dir

    for name in os.listdir(local_dir):
        if name.lower().endswith(".zip"):
            zip_path = os.path.join(local_dir, name)
            print(f"Extracting {name} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(local_dir)

    open(extracted_marker, "w").close()
    return local_dir


def find_class_folder(root: str, class_name: str):
    """UCF101 zip mirrors sometimes nest videos under an extra folder level."""
    for dirpath, dirnames, _ in os.walk(root):
        if class_name in dirnames:
            return os.path.join(dirpath, class_name)
    return None


GROUP_PATTERN = re.compile(r"_g(\d+)_c\d+", re.IGNORECASE)


def group_id(filename: str) -> str:
    """
    UCF101 filenames look like v_JugglingBalls_g01_c01.avi — clips sharing
    the same g## come from the same source recording (same actor/background).
    We must keep every clip from a group together in one split, or the
    model can "recognize the scene" instead of the action, inflating
    accuracy through leakage.
    """
    match = GROUP_PATTERN.search(filename)
    return match.group(1) if match else filename  # fallback: treat as its own group


def build_subset(root: str) -> None:
    random.seed(SEED)

    for class_name in CLASSES:
        src_folder = find_class_folder(root, class_name)
        if src_folder is None:
            print(f"[WARN] Could not find class folder for '{class_name}' — skipping.")
            continue

        videos = [f for f in os.listdir(src_folder) if f.lower().endswith((".avi", ".mp4"))]

        # Bucket clips by source group, then split at the GROUP level so no
        # group's clips end up split across train and test.
        groups = {}
        for v in videos:
            groups.setdefault(group_id(v), []).append(v)

        group_ids = list(groups.keys())
        random.shuffle(group_ids)
        split_idx = max(1, int(len(group_ids) * TRAIN_SPLIT))
        train_group_ids, test_group_ids = group_ids[:split_idx], group_ids[split_idx:]

        train_videos = [v for g in train_group_ids for v in groups[g]]
        test_videos = [v for g in test_group_ids for v in groups[g]]

        for split_name, split_videos in [("train", train_videos), ("test", test_videos)]:
            dst_folder = os.path.join(DATA_DIR, split_name, class_name)
            os.makedirs(dst_folder, exist_ok=True)
            for v in split_videos:
                shutil.copy(os.path.join(src_folder, v), os.path.join(dst_folder, v))

        print(f"{class_name}: {len(train_videos)} train / {len(test_videos)} test clips")

    print(f"\nDone. Subset written to: {DATA_DIR}")


if __name__ == "__main__":
    local_dir = download_full_dataset()
    extracted_root = extract_if_needed(local_dir)
    build_subset(extracted_root)
