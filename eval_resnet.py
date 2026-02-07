"""
eval_resnet.py — Standalone Evaluation Script
==============================================
Loads the saved resnet50_ecg_best.keras model and regenerates:
  - Val accuracy / loss
  - Confusion matrix (raw + normalized)
  - Per-class precision / recall / F1 bar chart
  - ROC-AUC curves
  - Summary dashboard
  - classification_report.txt

Run after training:
    python eval_resnet.py
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split as sk_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

warnings.filterwarnings('ignore')

# ── Register FocalLoss so Keras can deserialize the saved model ───────────────
@tf.keras.utils.register_keras_serializable(package='ECG')
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n_cls        = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_s     = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / n_cls)
        ce           = tf.keras.losses.categorical_crossentropy(y_true_s, y_pred)
        p_t          = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        return focal_weight * ce

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return cfg

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path('data')
MODEL_DIR    = Path('resnet_models')
GRAPH_DIR    = MODEL_DIR / 'graphs'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
SEED         = 42

class_names  = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"Classes: {class_names}")

# ── Recreate the same val split as training ───────────────────────────────────
all_files, all_labels = [], []
for cls_idx, cls in enumerate(class_names):
    imgs = sorted((DATA_DIR / cls).glob('*.jpg'))
    all_files.extend(imgs)
    all_labels.extend([cls_idx] * len(imgs))

_, val_files, _, val_labels = sk_split(
    all_files, all_labels,
    test_size=0.2, stratify=all_labels, random_state=SEED
)
print(f"Val samples: {len(val_files)}")

# Write val images to a temp dir
VAL_SPLIT = Path('_eval_tmp') / 'val'
import shutil
if VAL_SPLIT.exists():
    shutil.rmtree(VAL_SPLIT.parent)
for f, l in zip(val_files, val_labels):
    dst = VAL_SPLIT / class_names[l]
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dst / f.name)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = val_datagen.flow_from_directory(
    str(VAL_SPLIT), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False, seed=SEED, classes=class_names
)

# ── Load model ────────────────────────────────────────────────────────────────
best_model_path = str(MODEL_DIR / 'resnet50_ecg_best.keras')
print(f"\nLoading model from {best_model_path} …")
model = tf.keras.models.load_model(best_model_path, compile=False)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
    loss=FocalLoss(gamma=2.0, label_smoothing=0.05),
    metrics=['accuracy']
)
print("Model loaded ✓")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[STEP 1] Evaluating on validation set …")
val_gen.reset()
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
print(f"  Val Loss: {val_loss:.4f}  |  Val Accuracy: {val_acc:.4f}")

# ── Predictions ───────────────────────────────────────────────────────────────
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = val_gen.classes

# ── Confusion Matrix ──────────────────────────────────────────────────────────
print("\n[STEP 2] Generating confusion matrices …")
cm     = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, fmt, title in [
    (axes[0], cm,      'd',     'Confusion Matrix (Counts)'),
    (axes[1], cm_norm, '.2f',   'Confusion Matrix (Normalized)')
]:
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, cbar=True)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'confusion_matrix.png',            dpi=200, bbox_inches='tight')
plt.savefig(GRAPH_DIR / 'confusion_matrix_normalized.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Confusion matrices saved.")

# ── Classification Report ─────────────────────────────────────────────────────
print("\n[STEP 3] Classification report …")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)
with open(MODEL_DIR / 'classification_report.txt', 'w') as f:
    f.write(f"Val Accuracy : {val_acc:.4f}\nVal Loss     : {val_loss:.4f}\n\n")
    f.write(report)

# ── Per-class metrics bar chart ───────────────────────────────────────────────
from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))

x = np.arange(len(class_names)); w = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - w, prec, w, label='Precision', color='#2196F3', alpha=0.85)
ax.bar(x,     rec,  w, label='Recall',    color='#4CAF50', alpha=0.85)
ax.bar(x + w, f1,   w, label='F1-Score',  color='#FF9800', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0, 1.1); ax.set_ylabel('Score'); ax.legend()
ax.set_title('Per-Class Precision / Recall / F1', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bars in [ax.patches[i::3*len(class_names)//3] for i in range(3)]:
    pass
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'per_class_metrics.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Per-class metrics bar chart saved.")

# ── ROC-AUC ───────────────────────────────────────────────────────────────────
print("\n[STEP 4] Generating ROC-AUC curves …")
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']

fig, ax = plt.subplots(figsize=(10, 7))
for i, (cls, col) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-AUC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'roc_auc_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ROC-AUC curves saved.")

# ── Summary Dashboard ─────────────────────────────────────────────────────────
print("\n[STEP 5] Generating summary dashboard …")
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax1,
            xticklabels=class_names, yticklabels=class_names, linewidths=0.5, cbar=False)
ax1.set_title('Normalized Confusion Matrix', fontweight='bold')
ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
ax1.tick_params(axis='x', rotation=20)

# 2. Per-class F1
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(class_names, f1, color=colors)
ax2.set_ylim(0, 1.1); ax2.set_ylabel('F1-Score')
ax2.set_title('Per-Class F1-Score', fontweight='bold')
ax2.tick_params(axis='x', rotation=15)
for bar, v in zip(bars, f1):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

# 3. ROC
ax3 = fig.add_subplot(gs[0, 2])
for i, (cls, col) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    ax3.plot(fpr, tpr, color=col, lw=2, label=f'{cls} ({auc(fpr,tpr):.2f})')
ax3.plot([0,1],[0,1],'k--',lw=1)
ax3.set_title('ROC-AUC', fontweight='bold'); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# 4. Summary text
ax4 = fig.add_subplot(gs[1, :])
ax4.axis('off')
summary_lines = [
    f"Best Model:  resnet_models/resnet50_ecg_best.keras",
    f"Val Accuracy: {val_acc:.4f}  ({val_acc*100:.2f}%)    |    Val Loss: {val_loss:.4f}",
    "",
    f"{'Class':<14}{'Precision':>12}{'Recall':>10}{'F1':>10}",
    "─" * 48,
]
for cls, p, r, f in zip(class_names, prec, rec, f1):
    summary_lines.append(f"{cls:<14}{p:>12.4f}{r:>10.4f}{f:>10.4f}")

ax4.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle('ResNet50 ECG Classification — Evaluation Summary', fontsize=15, fontweight='bold')
plt.savefig(GRAPH_DIR / 'summary_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Summary dashboard saved.")

# ── Cleanup temp dir ──────────────────────────────────────────────────────────
shutil.rmtree('_eval_tmp', ignore_errors=True)

print("\n" + "="*60)
print("  EVALUATION COMPLETE")
print("="*60)
print(f"  Val Accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")
print(f"  Val Loss     : {val_loss:.4f}")
print(f"  Graphs saved : {GRAPH_DIR}/")
print("="*60)