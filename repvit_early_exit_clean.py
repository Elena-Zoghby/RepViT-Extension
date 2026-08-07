# =============================================================================
# RepViT Early-Exit: Reproduction + Extension
# Clean consolidated notebook — paste each section as its own Colab cell
# =============================================================================

# %% [Cell 1] Setup — clone repo, install deps
# -----------------------------------------------------------------------------
"""
!git clone https://github.com/THU-MIG/RepViT.git
%cd RepViT
!pip install timm
"""

# %% [Cell 2] Imports
# -----------------------------------------------------------------------------
import os
import time
import json
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %% [Cell 3] Load pretrained RepViT-M1.5 (no distillation)
# -----------------------------------------------------------------------------
from model.repvit import repvit_m1_5

model = repvit_m1_5()  # distillation=False by default — this is what we use

"""
!wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_distill_300e.pth
"""

checkpoint = torch.load("repvit_m1_5_distill_300e.pth", map_location="cpu")
state_dict = checkpoint["model"]

msg = model.load_state_dict(state_dict, strict=False)
print(msg)
# Expect unexpected_keys only for classifier_dist.* (distillation head we don't use) — this is fine.

model = model.to(device)

# Confirm block structure (sanity check — all should be RepViTBlock except index 0)
for i, b in enumerate(model.features):
    print(i, type(b).__name__)


# %% [Cell 4] Mount Drive + extract Tiny-ImageNet dataset
# -----------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

zip_path = "/content/drive/MyDrive/tiny-imagenet.zip"
extract_path = "/content/tiny-imagenet-200"

if not os.path.exists(extract_path):
    print("Extracting dataset...")
    os.system(f'unzip -q "{zip_path}" -d /content/')
    print("Extraction complete!")
else:
    print("Dataset already extracted!")

print(os.listdir(extract_path))


# %% [Cell 5] Reorganize val set into ImageFolder structure
# -----------------------------------------------------------------------------
dataset_path = "/content/tiny-imagenet-200/tiny-imagenet-200"
val_dir = os.path.join(dataset_path, "val")
images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

if os.path.exists(images_dir):
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_name, class_name = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, image_name)
            dst = os.path.join(class_dir, image_name)
            if os.path.exists(src):
                shutil.move(src, dst)
    shutil.rmtree(images_dir, ignore_errors=True)

print("Done")


# %% [Cell 6] Build datasets and dataloaders
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(dataset_path + "/train", transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print("Classes:", len(train_dataset.classes))
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))

# Freeze backbone — we only train exit heads / final head / gate on top of it
for param in model.parameters():
    param.requires_grad = False
model.eval()


# %% [Cell 7] Feature extraction helpers
# -----------------------------------------------------------------------------
# 5 checkpoints, tapped at these block indices (verify against Cell 3 printout,
# ideally against actual stage transitions -- see note below):
#   CP1 = block 16, CP2 = block 21, CP3 = block 26, CP4 = block 31, CP5 = block 37
# "Continue" path = full backbone through the LAST block (block 42), which is
# a separate, later point than CP5 -- CP5 is just another intermediate exit,
# not the final output.
#
# NOTE: these indices reflect your code's original checkpoint spacing, not
# necessarily RepViT's true architectural stage3/stage4 boundary. Print each
# block's repr (or check for stride-2/channel-change) to confirm where the
# real stage transition sits if that distinction matters for your writeup.
TAP_INDICES = [16, 21, 26, 31, 37]  # CP1, CP2, CP3, CP4, CP5
EXIT_TAP_IDX = 3                     # index into TAP_INDICES/exit_heads used for gating decision (CP4)

def extract_checkpoint_features(model, x):
    """Returns pooled features at each of the 5 tap points, the raw feature
    map at the gating tap (CP4) needed to continue past it, AND the true
    final pooled output -- all from a SINGLE forward pass, so everything is
    guaranteed to correspond to the same images in the same order."""
    outputs = []
    gate_map = None
    with torch.no_grad():
        for i, block in enumerate(model.features):
            x = block(x)
            if i in TAP_INDICES:
                pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
                outputs.append(pooled)
                if i == TAP_INDICES[EXIT_TAP_IDX]:
                    gate_map = x.clone()
        final_pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)  # x is now post-block-42
    return outputs, gate_map, final_pooled

def continue_to_final(model, gate_map, final_head):
    """Continue from the gating tap's feature map (CP4) through the rest of
    the backbone (including past CP5, all the way to the last block), then
    classify with OUR trained final_head (NOT model.classifier, which is
    still the original 1000-class ImageNet head)."""
    with torch.no_grad():
        x = gate_map
        for i in range(TAP_INDICES[EXIT_TAP_IDX] + 1, len(model.features)):
            x = model.features[i](x)
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        output = final_head(pooled)
    return output


# %% [Cell 8] Extract CP1-CP5 + true final output features for train set
# -----------------------------------------------------------------------------
# IMPORTANT: this must be a SINGLE pass over train_loader. train_loader has
# shuffle=True, so two separate loops over it produce different batch orders
# -- pairing features extracted in one loop with labels from another loop
# silently misaligns them (this caused final_head to train against
# effectively random labels earlier). One pass guarantees everything lines up.
checkpoint_features = [[] for _ in range(5)]
final_train_features = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        feats, _, final_pooled = extract_checkpoint_features(model, images)
        for i in range(5):
            checkpoint_features[i].append(feats[i].cpu())
        final_train_features.append(final_pooled.cpu())
        all_labels.append(labels)
        if batch_idx % 100 == 0:
            print("extraction batch", batch_idx)

checkpoint_features = [torch.cat(checkpoint_features[i], dim=0) for i in range(5)]
final_train_features = torch.cat(final_train_features, dim=0)
all_labels = torch.cat(all_labels, dim=0)


# %% [Cell 10] Build train/val split: CP1-CP5 features + true final features
# -----------------------------------------------------------------------------
NUM_CLASSES = len(train_dataset.classes)  # 200 for Tiny-ImageNet

# indices 0-4 = CP1..CP5 pooled features, index 5 = TRUE final output features,
# index 6 = labels
dataset_6feat = TensorDataset(
    checkpoint_features[0], checkpoint_features[1], checkpoint_features[2],
    checkpoint_features[3], checkpoint_features[4],
    final_train_features,
    all_labels
)
LABEL_IDX = 6

n_total = len(dataset_6feat)
n_val = int(0.1 * n_total)
n_train = n_total - n_val

train_data, val_data = random_split(
    dataset_6feat, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader_features = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader_features = DataLoader(val_data, batch_size=256, shuffle=False)


# %% [Cell 11] Exit head definition + training function
# -----------------------------------------------------------------------------
class ExitHead(nn.Module):
    """BN + Linear, matching RepViT's own BN_Linear classifier design.
    The BatchNorm matters most for deeper/final features, which tend to have
    large, badly-scaled magnitudes -- without it, training can stall near
    random (this is exactly what happened when final_head used a plain
    Linear with no BN: loss stuck near ln(num_classes), never dropping)."""
    def __init__(self, in_features, num_classes=NUM_CLASSES):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.classifier(self.bn(x))


def train_head(head_model, optimizer, feature_index, loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    head_model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            features = batch[feature_index].to(device)
            labels = batch[LABEL_IDX].to(device)

            optimizer.zero_grad()
            outputs = head_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(f"  Epoch {epoch+1}/{epochs}  Loss {total_loss/len(loader):.4f}  "
              f"Acc {100*correct/total:.2f}%")


# %% [Cell 12] Train exit heads for CP1-CP5, plus a SEPARATE final_head
# -----------------------------------------------------------------------------
# exit_heads[0..4]  -> trained on CP1..CP5 pooled features (5 intermediate exits)
# final_head        -> trained SEPARATELY on the true final output features
#                      (post block-42), used only for the "continue past CP4"
#                      branch. NOT the same as exit_heads[4] (CP5) -- CP5 is
#                      just another intermediate checkpoint, not the network's
#                      true final output.
CP_FEATURE_DIM = checkpoint_features[0].shape[1]   # dim of CP1-CP5 pooled features, read directly (was hardcoded before)
FINAL_FEATURE_DIM = model.classifier.classifier.l.in_features  # 512, confirmed from pretrained head

print("CP feature dim:", CP_FEATURE_DIM)
print("Final feature dim:", FINAL_FEATURE_DIM)

exit_heads = [ExitHead(CP_FEATURE_DIM).to(device) for _ in range(5)]
exit_optimizers = [torch.optim.AdamW(h.parameters(), lr=1e-3) for h in exit_heads]

for i in range(5):
    print(f"\nCheckpoint {i+1}")
    train_head(exit_heads[i], exit_optimizers[i], i, train_loader_features)

final_head = ExitHead(FINAL_FEATURE_DIM).to(device)
final_optimizer = torch.optim.AdamW(final_head.parameters(), lr=1e-3)
print("\nFinal head (true final output, post-CP5)")
train_head(final_head, final_optimizer, 5, train_loader_features)


# %% [Cell 13] Full validation-set feature extraction (for eval + threshold sweep)
# -----------------------------------------------------------------------------
# Gating decision happens at CP4 (block 31): exit with CP4's prediction, or
# continue through CP5 + the rest of the network to the TRUE final output,
# scored by final_head (not exit_heads[4], which is CP5's own intermediate head).
gate_pred_all = []      # CP4 prediction (used if we exit)
final_pred_all = []     # true final-output prediction (used if we continue)
gate_conf_all = []      # CP4 softmax confidence (used as the gating signal)
labels_all_eval = []

for head in exit_heads:
    head.eval()
final_head.eval()
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        cp_features, gate_map, _ = extract_checkpoint_features(model, images)
        gate_vector = cp_features[EXIT_TAP_IDX]  # CP4 pooled features

        gate_logits = exit_heads[EXIT_TAP_IDX](gate_vector)
        gate_pred = gate_logits.argmax(dim=1)
        gate_conf = F.softmax(gate_logits, dim=1).max(dim=1).values

        final_logits = continue_to_final(model, gate_map, final_head)
        final_pred = final_logits.argmax(dim=1)

        gate_pred_all.append(gate_pred.cpu())
        final_pred_all.append(final_pred.cpu())
        gate_conf_all.append(gate_conf.cpu())
        labels_all_eval.append(labels.cpu())

gate_pred_all = torch.cat(gate_pred_all)
final_pred_all = torch.cat(final_pred_all)
gate_conf_all = torch.cat(gate_conf_all)
labels_all_eval = torch.cat(labels_all_eval)

always_continue_acc = (final_pred_all == labels_all_eval).float().mean().item() * 100
always_exit_acc = (gate_pred_all == labels_all_eval).float().mean().item() * 100

print(f"Always-continue baseline (true final output): {always_continue_acc:.2f}%")
print(f"Always-exit baseline (CP4 only):               {always_exit_acc:.2f}%")

# Optional: check standalone accuracy of each of the 5 checkpoints
for i in range(5):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader_features:
            features = batch[i].to(device)
            labels = batch[LABEL_IDX].to(device)
            pred = exit_heads[i](features).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"CP{i+1} standalone accuracy: {100*correct/total:.2f}%")


# %% [Cell 14] Confidence-gated early-exit: threshold sweep
# -----------------------------------------------------------------------------
# Confidence gating beat a learned MLP gate in earlier experiments
# (AUROC 0.80 vs 0.449) — so we use softmax confidence directly, no extra
# trained gate model needed. Simpler AND better.

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for threshold in thresholds:
    exit_mask = gate_conf_all >= threshold

    pred = torch.where(exit_mask, gate_pred_all, final_pred_all)
    hybrid_acc = (pred == labels_all_eval).float().mean().item() * 100

    exit_count = exit_mask.sum().item()
    continue_count = (~exit_mask).sum().item()

    exit_acc = (gate_pred_all[exit_mask] == labels_all_eval[exit_mask]).float().mean().item() * 100 \
        if exit_count > 0 else float('nan')
    continue_acc = (final_pred_all[~exit_mask] == labels_all_eval[~exit_mask]).float().mean().item() * 100 \
        if continue_count > 0 else float('nan')

    exit_fraction = exit_count / len(labels_all_eval) * 100

    results.append({
        "threshold": threshold,
        "hybrid_acc": hybrid_acc,
        "exit_acc": exit_acc,
        "continue_acc": continue_acc,
        "exit_count": exit_count,
        "continue_count": continue_count,
        "exit_fraction": exit_fraction,
    })

    print(f"Threshold {threshold}: Hybrid {hybrid_acc:.2f}% | "
          f"Exit acc {exit_acc:.2f}% (n={exit_count}) | "
          f"Continue acc {continue_acc:.2f}% | Exit frac {exit_fraction:.1f}%")


# %% [Cell 15] AUROC check (confirms confidence signal is meaningful)
# -----------------------------------------------------------------------------
def compute_auroc(scores, labels):
    scores = scores.cpu().float()
    labels = labels.cpu().float()
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos, n_neg = pos_scores.numel(), neg_scores.numel()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    all_scores = torch.cat([pos_scores, neg_scores])
    ranks = torch.argsort(torch.argsort(all_scores)).float() + 1
    sum_ranks_pos = ranks[:n_pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc.item()

gate_correct_eval = (gate_pred_all == labels_all_eval).float()
conf_auc = compute_auroc(gate_conf_all, gate_correct_eval)
print("Softmax confidence AUROC:", conf_auc)


# %% [Cell 16] Plot: hybrid accuracy vs exit fraction
# -----------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

exit_fractions = [r["exit_fraction"] for r in results]
hybrid_accs = [r["hybrid_acc"] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(exit_fractions, hybrid_accs, marker='o', label="Confidence-gated hybrid accuracy", color='tab:blue')
plt.axhline(y=always_continue_acc, color='gray', linestyle='--',
            label=f"Always-continue baseline ({always_continue_acc:.2f}%)")
plt.axhline(y=always_exit_acc, color='lightgray', linestyle=':',
            label=f"Always-exit baseline ({always_exit_acc:.2f}%)")
plt.xlabel("Exit fraction (%)")
plt.ylabel("Accuracy (%)")
plt.title("Early-Exit Accuracy vs. Compute Savings (Confidence Gating)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/hybrid_accuracy_vs_exit_fraction.png", dpi=150, bbox_inches='tight')
plt.show()


# %% [Cell 17] Save all artifacts (organized to match repo structure)
# -----------------------------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

for i, head in enumerate(exit_heads):
    torch.save(head.state_dict(), f"checkpoints/exit_head_cp{i+1}.pth")
torch.save(final_head.state_dict(), "checkpoints/final_head_true_output.pth")

with open("results/results_summary.json", "w") as f:
    json.dump({
        "always_continue_baseline": always_continue_acc,
        "always_exit_baseline": always_exit_acc,
        "confidence_auc": conf_auc,
        "threshold_sweep": results,
    }, f, indent=2)

print("Saved: checkpoints/exit_head_cp1-5.pth, checkpoints/final_head_true_output.pth, "
      "results/results_summary.json, results/hybrid_accuracy_vs_exit_fraction.png")
