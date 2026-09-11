"""
Produces "sports-analysis style" annotated videos: takes a handful of
random test clips, overlays MediaPipe Pose skeleton joints on every
frame, and stamps the true vs. predicted action label — showing how the
model's decision lines up with the actual body movement in each clip.

This is the video-level equivalent of KOLSA's joint-marking visuals,
adapted to action classification instead of a juggling quality score.

Usage:
    python src/visualize_pose_predictions.py --checkpoint checkpoints/best_model.pt --num_clips 5

Output:
    outputs/pose_visualizations/<ClassName>_<n>_true-X_pred-Y.mp4
"""

import argparse
import os
import random
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, FRAME_SIZE, NUM_FRAMES, OUTPUT_DIR, SEED, get_classes  # noqa: E402
from src.dataset import MEAN, STD, load_clip  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def predict_clip(model, video_path: str, device, num_frames: int, frame_size: int) -> int:
    """Runs the classifier on a clip the same way evaluate.py does, to get a predicted label."""
    clip = load_clip(video_path, num_frames, frame_size, train=False).astype(np.float32) / 255.0
    clip = (clip - MEAN) / STD
    tensor = torch.from_numpy(clip.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return pred


def annotate_video(video_path: str, out_path: str, true_label: str, pred_label: str) -> None:
    """Draws pose skeleton + label banner on every frame, writes an annotated mp4."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    correct = true_label == pred_label
    banner_color = (0, 200, 0) if correct else (0, 0, 220)  # BGR: green if correct, red if wrong

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            # Label banner at the top of the frame
            banner_h = 46
            cv2.rectangle(frame, (0, 0), (width, banner_h), (30, 30, 30), thickness=-1)
            text = f"true: {true_label}   pred: {pred_label}"
            cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)

            writer.write(frame)

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model.pt"))
    parser.add_argument("--num_clips", type=int, default=5)
    args = parser.parse_args()

    random.seed(SEED)
    device = get_device()

    out_dir = os.path.join(OUTPUT_DIR, "pose_visualizations")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get("classes", get_classes())

    model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_root = os.path.join(DATA_DIR, "test")
    all_clips = []
    for class_name in classes:
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".avi", ".mp4")):
                all_clips.append((os.path.join(class_dir, fname), class_name))

    if not all_clips:
        raise RuntimeError(f"No test clips found under {test_root}")

    chosen = random.sample(all_clips, min(args.num_clips, len(all_clips)))

    for i, (video_path, true_label) in enumerate(chosen, start=1):
        pred_idx = predict_clip(model, video_path, device, NUM_FRAMES, FRAME_SIZE)
        pred_label = classes[pred_idx]

        status = "correct" if pred_label == true_label else "WRONG"
        out_name = f"{i:02d}_{true_label}_true-{true_label}_pred-{pred_label}.mp4"
        out_path = os.path.join(out_dir, out_name)

        print(f"[{i}/{len(chosen)}] {os.path.basename(video_path)} "
              f"| true: {true_label} | pred: {pred_label} | {status}")
        annotate_video(video_path, out_path, true_label, pred_label)

    print(f"\nSaved {len(chosen)} pose-annotated videos to: {out_dir}")


if __name__ == "__main__":
    main()
