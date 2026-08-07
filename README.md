# RepViT-Extension
## Confidence-Gated Early Exit for RepViT

Reproduction of RepViT-M1.5 inference on Tiny-ImageNet, extended with a confidence-gated early-exit mechanism that lets "easy" inputs skip stage 4 of the network.

**Full writeup:** see [`project_report.docx`](./project_report.docx) for the complete methodology, results, and analysis.

## Summary

- **Reproduction:** Loaded pretrained RepViT-M1.5 (ImageNet-1k, non-distilled), adapted and fine-tuned a new classification head for Tiny-ImageNet (200 classes). Full-model baseline accuracy: **80.40%**.
- **Extension:** Added 5 lightweight exit heads (CP1–CP5) tapped throughout stage 3 (blocks 16, 21, 26, 31, 37). At CP4, a gate decides whether to exit early or continue through stage 4.
- **Key finding:** A learned MLP gate failed (AUROC 0.449, worse than random). A simple softmax-confidence gate worked well instead (AUROC 0.837).
- **Result:** At the best threshold (0.9), hybrid accuracy is **80.34%** — within 0.06 points of the full-model baseline — while skipping stage 4 entirely for **16.9%** of inputs.

## Repo structure

```
.
├── repvit_early_exit_clean.py   # Full pipeline: setup → data → training → evaluation
├── project_report.docx          # 2-4 page project report
├── results/
│   ├── results_summary.json     # Threshold sweep, baselines, AUROC
│   └── hybrid_accuracy_vs_exit_fraction.png
└── README.md
```

> **Note:** Trained checkpoints (`exit_head_cp*.pth`, `final_head_true_output.pth`) are not committed to this repo to keep it lightweight — they're small and fast to reproduce by rerunning the notebook (see below). The base RepViT-M1.5 pretrained weights are downloaded directly from the [official RepViT release](https://github.com/THU-MIG/RepViT/releases) in Cell 3.

## How to run

This was developed and run on Google Colab (GPU runtime). To reproduce:

1. Open a fresh Colab notebook.
2. Paste each `# %% [Cell N]` block from `repvit_early_exit_clean.py` into its own cell, in order.
3. Mount your Google Drive and point `zip_path` in Cell 4 at a `tiny-imagenet.zip` (download from the [Tiny-ImageNet dataset page](https://www.kaggle.com/c/tiny-imagenet)).
4. Run all cells top to bottom.
5. Commit the regenerated `results/` folder (small JSON + PNG) if it changes — everything else (`checkpoints/`, the dataset) stays local and is gitignored.

Runtime: full feature extraction over the Tiny-ImageNet train set (~100k images) takes the bulk of the time; head training itself is fast (a few minutes per checkpoint).

## Method details

**Base model:** [RepViT-M1.5](https://github.com/THU-MIG/RepViT) (CVPR 2024), pretrained on ImageNet-1k, frozen throughout — only new heads are trained.

**Architecture verification:** RepViT-M1.5's backbone is 43 sequential blocks. Stage boundaries were confirmed directly from the model (stride-2 downsample + channel-count jump), not assumed:

| Stage | Blocks | Channels |
|---|---|---|
| Stem | 0 | 32→64 |
| Stage 1 | 1–5 | 64 |
| Stage 2 | 6–11 | 128 |
| Stage 3 | 12–37 | 256 |
| Stage 4 | 38–42 | 512 |

**Exit heads:** Each is a `BatchNorm1d + Linear` (matching RepViT's own classifier design), trained on pooled features from its tap point.

**Gating:** At CP4 (block 31, inside stage 3), the gate compares the CP4 head's softmax confidence to a threshold. Above threshold → exit with CP4's prediction. Below threshold → continue through the rest of the network to a separately-trained final head.

## Results

| Checkpoint | Depth (block) | Standalone Accuracy |
|---|---|---|
| CP1 | 16 | 38.46% |
| CP2 | 21 | 44.58% |
| CP3 | 26 | 52.20% |
| CP4 | 31 | 58.84% |
| CP5 | 37 | 71.00% |
| Full model | 42 | 80.40% |

| Threshold | Hybrid Acc | Exit Acc | Continue Acc | Exit Fraction |
|---|---|---|---|---|
| 0.3 | 67.91% | 69.44% | 62.85% | 76.8% |
| 0.5 | 75.82% | 82.09% | 68.84% | 52.7% |
| 0.7 | 79.27% | 91.47% | 72.91% | 34.2% |
| 0.8 | 79.91% | 94.65% | 74.78% | 25.8% |
| 0.9 | **80.34%** | 97.09% | 76.94% | **16.9%** |

See `project_report.docx` for full discussion, limitations, and the debugging story behind these numbers.

## Acknowledgments

- [RepViT](https://github.com/THU-MIG/RepViT) (Wang et al., CVPR 2024) — original model and pretrained checkpoint.
- [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet) — evaluation dataset.
