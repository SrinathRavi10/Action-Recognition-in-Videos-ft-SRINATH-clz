# Action Recognition in Videos — UCF101

A deep-learning system that classifies short video clips into human action
categories (e.g. jump rope, basketball, push-ups) using a fine-tuned 3D
CNN, evaluated on a 10-class subset of the UCF101 benchmark.

## Project Description

This project builds a video action classifier: given a short clip, the
model predicts which of a set of known actions is being performed. It
covers the full pipeline required for real-world video understanding —
preprocessing raw video into model-ready clips, training a deep network
on labeled action classes, and evaluating it with standard classification
metrics (top-1/top-5 accuracy, confusion matrix, per-class precision/recall/F1,
and inference latency).

The domain is sports and everyday-movement analytics — the same space as
automated performance feedback, surveillance/behavior analysis, and video
indexing described in the project brief.

## Approach

**Backbone: R(2+1)D-18, pretrained on Kinetics-400, fine-tuned on UCF101.**

Rather than training a 3D CNN from scratch (data- and compute-hungry),
this project uses transfer learning:

1. **Preprocessing** — each video is decoded with OpenCV, and 16 frames
   are sampled evenly across its duration (with light random temporal
   jitter during training), resized to 112×112, and normalized using the
   same statistics the backbone was pretrained with.
2. **Backbone** — [`r2plus1d_18`](https://arxiv.org/abs/1711.11248)
   factorizes 3D convolutions into a 2D spatial + 1D temporal convolution.
   This keeps memory and compute low enough to fine-tune comfortably on a
   single GPU with ≤16GB memory (including a free Colab GPU), while still
   capturing motion dynamics across frames — unlike frame-independent 2D
   CNNs.
3. **Transfer learning strategy** — the backbone is loaded with
   Kinetics-400 pretrained weights, most layers are frozen, and only the
   last residual block plus a new final classification layer are
   fine-tuned on our class subset. This drastically reduces the amount of
   labeled data and training time needed to reach good accuracy.
4. **Augmentation** — random horizontal flip and a mild random crop +
   resize, applied per-clip during training only.
5. **Training** — AdamW optimizer, cosine learning-rate schedule,
   mixed-precision (`torch.cuda.amp`) to fit larger batches in limited
   GPU memory, cross-entropy loss, fixed random seeds for reproducibility.
6. **Evaluation** — top-1/top-5 accuracy, a confusion matrix heatmap,
   a full per-class precision/recall/F1 report, and average per-clip
   inference time, all written to `outputs/`.
7. **Class subset** — starting with 10 UCF101 classes
   (`JugglingBalls`, `JumpRope`, `JumpingJack`, `WalkingWithDog`,
   `Basketball`, `TennisSwing`, `GolfSwing`, `PushUps`, `PullUps`,
   `Bowling`) rather than the full 101, to keep training feasible on
   limited hardware. The class list is a single line to edit in
   `src/config.py` if you want to scale up.

## Repository Structure

```
.
├── data/
│   └── download_ucf101.py      # downloads UCF101 + builds the class subset/split
├── src/
│   ├── config.py                # classes, paths, hyperparameters
│   ├── dataset.py                # video loading, frame sampling, augmentation
│   ├── model.py                  # R(2+1)D-18 backbone + classifier head
│   ├── train.py                  # training loop + checkpointing
│   ├── evaluate.py                # accuracy / confusion matrix / report / timing
│   ├── visualize_predictions.py  # qualitative prediction grid on sample clips
│   └── utils.py                   # seeding, device helpers
├── notebooks/
│   └── Action_Recognition_Colab.ipynb  # end-to-end run in Google Colab
├── requirements.txt
└── .gitignore
```

## Prerequisites / Dependencies

- Python 3.9+
- A CUDA-capable GPU with ≤16GB memory is recommended (CPU also works, just slower)
- See `requirements.txt` for exact package versions:
  - `torch`, `torchvision` — model + training
  - `opencv-python` — video decoding
  - `numpy`, `scikit-learn` — numeric ops, evaluation metrics
  - `matplotlib`, `seaborn` — plots
  - `tqdm` — progress bars
  - `huggingface_hub` — dataset download

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Download the dataset and build the class subset:**
```bash
python data/download_ucf101.py
```
This pulls UCF101 from a Hugging Face mirror and writes only the
configured classes into `data/UCF101_subset/{train,test}/<ClassName>/`.

**2. Train:**
```bash
python src/train.py
```
Checkpoints the best-validation-accuracy model to `checkpoints/best_model.pt`.

**3. Evaluate:**
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```
Prints top-1/top-5 accuracy and inference time, and writes
`outputs/confusion_matrix.png` + `outputs/classification_report.txt`.

**4. Visualize predictions on sample clips:**
```bash
python src/visualize_predictions.py --checkpoint checkpoints/best_model.pt --num_samples 8
```
Writes `outputs/sample_predictions.png` — a grid of sample clips with
true vs. predicted labels.

**Or run everything in Colab:** open `notebooks/Action_Recognition_Colab.ipynb`,
which installs dependencies, downloads the data, trains, and evaluates
end-to-end on a free Colab GPU.

## Results

_Fill in after training:_

| Metric | Value |
|---|---|
| Top-1 accuracy | — |
| Top-5 accuracy | — |
| Avg inference time / clip | — ms |

See `outputs/classification_report.txt` and `outputs/confusion_matrix.png`
for the full per-class breakdown.

## Possible Extensions

- Scale from 10 to the full 101 UCF101 classes
- Add optical-flow input as a second stream (Two-Stream Network)
- Try a transformer-based backbone (e.g. TimeSformer, Video Swin) once
  the CNN baseline is stable
- Evaluate on HMDB51 to test cross-dataset generalization
