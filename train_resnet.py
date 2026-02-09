"""
ECG Heart Disease Detection — ResNet50 Training Script
=======================================================
Features:
  - Dataset analysis & class count visualization
  - Dataset balancing via augmented oversampling
  - ResNet50 backbone (ImageNet weights)
  - Two-phase training: frozen base → partial fine-tuning
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
  - Output graphs:
        1. Class Distribution (before & after balancing)
        2. Train vs Val Accuracy
        3. Train vs Val Loss
        4. Fine-tune Accuracy
        5. Fine-tune Loss
        6. Combined Accuracy (both phases)
        7. Combined Loss (both phases)
        8. Confusion Matrix (heatmap)
        9. Normalized Confusion Matrix
       10. Per-class Precision / Recall / F1 bar chart
       11. ROC-AUC curves (one-vs-rest, per class)
  - Saves model as  resnet_models/resnet50_ecg_best.keras
  - Saves class_names.txt
"""

import os, shutil, random, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
)

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR      = Path('data')
BALANCED_DIR  = Path('data_balanced')    # temporary balanced dataset
MODEL_DIR     = Path('resnet_models')
GRAPH_DIR     = MODEL_DIR / 'graphs'

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
PHASE1_EPOCHS = 30    # frozen backbone
PHASE2_EPOCHS = 50    # fine-tune last 50 layers — Phase 2 was still improving at epoch 30
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  ECG ResNet50 Training  —  Starting up")
print("=" * 60)

# ── 1. Dataset Analysis ───────────────────────────────────────────────────────
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"\nClasses found: {class_names}")

counts_before = {c: len(list((DATA_DIR / c).glob('*.jpg'))) for c in class_names}
print("\nClass distribution (original):")
for c, n in counts_before.items():
    print(f"  {c:12s}: {n}")

# Save class names
with open('class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))
print("\nSaved class_names.txt")

# ── 2. Stratified Train/Val Split on ORIGINAL data ───────────────────────────
# KEY FIX: Val must use only original images — NOT the balanced dir which
# contains augmented copies the model has effectively "seen" during training.
print("\n[STEP 2] Stratified 80/20 split on original images …")

from sklearn.model_selection import train_test_split as sk_split

all_files, all_labels = [], []
for cls_idx, cls in enumerate(class_names):
    imgs = sorted((DATA_DIR / cls).glob('*.jpg'))
    all_files.extend(imgs)
    all_labels.extend([cls_idx] * len(imgs))

train_files, val_files, train_labels, val_labels = sk_split(
    all_files, all_labels,
    test_size=0.2, stratify=all_labels, random_state=SEED
)
print(f"  Train originals: {len(train_files)}  |  Val originals: {len(val_files)}")

# ── 3. Oversample Training Split Only ─────────────────────────────────────────
# Validation stays 100% original — no augmented images leak into eval
print("\n[STEP 3] Oversampling training split to balance classes …")

aug_gen_bal = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    horizontal_flip=False,
    brightness_range=[0.85, 1.15],
    fill_mode='reflect'
)

if BALANCED_DIR.exists():
    shutil.rmtree(BALANCED_DIR)

TRAIN_SPLIT = BALANCED_DIR / 'train'
VAL_SPLIT   = BALANCED_DIR / 'val'

# Group training files by class
train_by_cls = {cls: [] for cls in class_names}
for f, l in zip(train_files, train_labels):
    train_by_cls[class_names[l]].append(f)

TARGET       = max(len(v) for v in train_by_cls.values())
counts_after = {}

for cls in class_names:
    cls_imgs = train_by_cls[cls]
    dst = TRAIN_SPLIT / cls
    dst.mkdir(parents=True, exist_ok=True)

    for img_path in cls_imgs:                     # copy originals
        shutil.copy(img_path, dst / img_path.name)

    counter = 0
    extra   = TARGET - len(cls_imgs)
    while counter < extra:
        src_img = random.choice(cls_imgs)
        img_arr = img_to_array(load_img(src_img, target_size=IMG_SIZE))
        x       = img_arr.reshape((1,) + img_arr.shape)
        for batch in aug_gen_bal.flow(x, batch_size=1, seed=SEED + counter):
            tf.keras.utils.save_img(str(dst / f"aug_{counter:05d}.jpg"),
                                    batch[0].astype('uint8'))
            counter += 1
            break
    counts_after[cls] = len(list(dst.glob('*.jpg')))
    print(f"  {cls:12s}: {len(cls_imgs):4d} orig  →  {counts_after[cls]:4d} balanced")

# Copy val originals (no augmentation, no duplicates)
for f, l in zip(val_files, val_labels):
    dst = VAL_SPLIT / class_names[l]
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dst / f.name)

counts_val = {cls: len(list((VAL_SPLIT / cls).glob('*.jpg'))) for cls in class_names}
print(f"  Validation per class (originals only): {counts_val}")

# ── 3b. Class distribution plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']

axes[0].bar(list(counts_before.keys()), list(counts_before.values()), color=colors)
axes[0].set_title('Class Distribution — Original', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Class'); axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(axes[0].patches, counts_before.values()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha='center', fontsize=11, fontweight='bold')

axes[1].bar(list(counts_after.keys()), list(counts_after.values()), color=colors)
axes[1].set_title('Training Distribution — After Balancing', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Class'); axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(axes[1].patches, counts_after.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Dataset Balancing (Val = Original Images Only)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Graph saved: class_distribution.png")

# ── 4. Data Generators ────────────────────────────────────────────────────────
print("\n[STEP 4] Creating data generators …")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    brightness_range=[0.85, 1.15],
    fill_mode='reflect',
    shear_range=0.05
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    str(TRAIN_SPLIT), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=SEED, classes=class_names
)
val_gen = val_datagen.flow_from_directory(
    str(VAL_SPLIT), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False, seed=SEED, classes=class_names
)

print(f"  Training samples  : {train_gen.n}  (balanced + augmented)")
print(f"  Validation samples: {val_gen.n}  (original only)")
print(f"  Class indices     : {train_gen.class_indices}")

# Class weights from original training split labels
# compute_class_weight already gives post_mi the highest weight
# (it's the smallest class). No manual boost needed — boosting 2x caused
# the model to over-predict post_mi for every ambiguous image in production.
y_train_orig = np.array(train_labels)
cw = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=y_train_orig)
class_weights = dict(enumerate(cw))
print(f"  Class weights (balanced): { {class_names[k]: round(v, 3) for k, v in class_weights.items()} }")

# ── 4. Build ResNet50 Model ───────────────────────────────────────────────────
print("\n[STEP 4] Building ResNet50 model …")

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False    # Phase 1: frozen backbone

# L2 regularizer for dense layers to control overfitting
reg = tf.keras.regularizers.l2(1e-4)

x = base.output
x = layers.GlobalAveragePooling2D(name='gap')(x)
x = layers.BatchNormalization(name='bn_head')(x)
x = layers.Dense(512, activation='relu', name='dense_1', kernel_regularizer=reg)(x)
x = layers.Dropout(0.5, name='drop_1')(x)          # increased from 0.4
x = layers.Dense(256, activation='relu', name='dense_2', kernel_regularizer=reg)(x)
x = layers.Dropout(0.4, name='drop_2')(x)          # increased from 0.3
outputs = layers.Dense(len(class_names), activation='softmax', name='predictions')(x)

model = Model(inputs=base.input, outputs=outputs, name='ResNet50_ECG')

# Phase 1: Standard CE (no label smoothing) so the head converges fast
# Label smoothing in Phase 1 was the reason val_acc was stuck at 50.9%
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\n  Trainable params (Phase 1): {sum(p.numpy().size for p in model.trainable_weights):,}")

# ── 5. Phase 1 Training (Frozen Backbone) ────────────────────────────────────
print("\n[STEP 6] Phase 1 — Training with frozen backbone …")

best_model_path = str(MODEL_DIR / 'resnet50_ecg_best.keras')

callbacks_p1 = [
    ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    # Monitor val_accuracy (same as checkpoint) so EarlyStopping restores the same best epoch
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                  mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7, verbose=1),
    CSVLogger(str(MODEL_DIR / 'phase1_log.csv'), append=False)
]

t0 = time.time()
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_p1,
    verbose=1
)
print(f"  Phase 1 done in {(time.time()-t0)/60:.1f} min")

# ── 6. Phase 2 Training (Fine-tuning) ────────────────────────────────────────
print("\n[STEP 7] Phase 2 — Fine-tuning last 50 layers …")

# IMPORTANT: Load the best checkpoint (best val_accuracy) before fine-tuning.
# model.layers works on the flat layer list, so no get_layer('resnet50') needed.
model = tf.keras.models.load_model(best_model_path)

# The custom head is the last 7 layers: gap, bn_head, dense_1, drop_1, dense_2, drop_2, predictions
# Everything before them is the ResNet50 backbone.
HEAD_LAYERS = 7
backbone_layers = model.layers[:-HEAD_LAYERS]

# Unfreeze all backbone first, then re-freeze all but the last 50 layers
for layer in backbone_layers:
    layer.trainable = True
for layer in backbone_layers[:-50]:
    layer.trainable = False

fine_tune_count = sum(1 for l in backbone_layers if l.trainable)
print(f"  Unfrozen backbone layers : {fine_tune_count}")
print(f"  Trainable params (Phase 2): {sum(p.numpy().size for p in model.trainable_weights):,}")

# Phase 2: Focal loss (gamma=2) focuses on hard-to-classify examples like post_mi.
# Implemented as a registered Keras Loss subclass so the model can be saved/loaded.
@tf.keras.utils.register_keras_serializable(package='ECG')
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma           = gamma
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

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss=FocalLoss(gamma=2.0, label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    # Monitor val_accuracy to match checkpoint and avoid restoring wrong epoch
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                  mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, min_lr=1e-8, verbose=1),
    CSVLogger(str(MODEL_DIR / 'phase2_log.csv'), append=False)
]

t0 = time.time()
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_p2,
    verbose=1
)
print(f"  Phase 2 done in {(time.time()-t0)/60:.1f} min")

# Re-load the best model and recompile for evaluation (model.evaluate needs compiled loss)
model = tf.keras.models.load_model(best_model_path, compile=False)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss=FocalLoss(gamma=2.0, label_smoothing=0.05),
    metrics=['accuracy']
)
print(f"\n  Best model saved at: {best_model_path}")

# ── 7.  Plot Training Curves ─────────────────────────────────────────────────
print("\n[STEP 7] Generating training curve graphs …")

def plot_metric(history, metric, title, save_path, phase_label=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history[metric],
            label=f'Train {metric.replace("_"," ").title()}', color='#2196F3', marker='o', markersize=4)
    ax.plot(history.history[f'val_{metric}'],
            label=f'Val {metric.replace("_"," ").title()}', color='#FF9800', marker='s', markersize=4)
    ax.set_title(f'{phase_label}{title}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel(metric.replace("_"," ").title())
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# Phase 1 curves
plot_metric(history1, 'accuracy', 'Accuracy',  GRAPH_DIR / 'phase1_accuracy.png', 'Phase 1 — ')
plot_metric(history1, 'loss',     'Loss',       GRAPH_DIR / 'phase1_loss.png',     'Phase 1 — ')

# Phase 2 curves
plot_metric(history2, 'accuracy', 'Accuracy',  GRAPH_DIR / 'phase2_accuracy.png', 'Fine-tune — ')
plot_metric(history2, 'loss',     'Loss',       GRAPH_DIR / 'phase2_loss.png',     'Fine-tune — ')

# Combined curves (both phases concatenated)
def combine_hist(h1, h2, key):
    return h1.history[key] + h2.history[key]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ep1 = len(history1.history['accuracy'])
ep_total = ep1 + len(history2.history['accuracy'])
x_p1 = range(1, ep1 + 1)
x_p2 = range(ep1 + 1, ep_total + 1)

for ax, metric, ylabel in [(ax1, 'accuracy', 'Accuracy'), (ax2, 'loss', 'Loss')]:
    ax.plot(x_p1, history1.history[metric],         color='#2196F3', marker='o', markersize=3, label='Train (Phase 1)')
    ax.plot(x_p1, history1.history[f'val_{metric}'],color='#90CAF9', marker='s', markersize=3, label='Val (Phase 1)')
    ax.plot(x_p2, history2.history[metric],         color='#E91E63', marker='o', markersize=3, label='Train (Fine-tune)')
    ax.plot(x_p2, history2.history[f'val_{metric}'],color='#F48FB1', marker='s', markersize=3, label='Val (Fine-tune)')
    ax.axvline(x=ep1, color='gray', linestyle='--', linewidth=1.5, label='Fine-tune starts')
    ax.set_title(f'Combined {ylabel}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Full Training History (Phase 1 + Fine-tuning)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'combined_training_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Training curve graphs saved.")

# ── 8. Evaluate on Validation Set ────────────────────────────────────────────
print("\n[STEP 8] Evaluating on validation set …")

val_gen.reset()
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
print(f"  Val Loss: {val_loss:.4f}  |  Val Accuracy: {val_acc:.4f}")

# Get predictions
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes[:len(y_pred)]

# ── 9. Confusion Matrix ───────────────────────────────────────────────────────
print("\n[STEP 9] Generating confusion matrices …")

def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False, fmt='.2f'):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=10)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:{fmt}}", ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black', fontsize=11)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, class_names, 'Confusion Matrix', GRAPH_DIR / 'confusion_matrix.png', fmt='d')
plot_confusion_matrix(cm, class_names, 'Normalized Confusion Matrix',
                      GRAPH_DIR / 'confusion_matrix_normalized.png', normalize=True)
print("  Confusion matrices saved.")

# ── 10. Classification Report ─────────────────────────────────────────────────
print("\n[STEP 10] Classification report …")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Save text report
with open(MODEL_DIR / 'classification_report.txt', 'w') as f:
    f.write(f"Validation Loss : {val_loss:.4f}\n")
    f.write(f"Validation Acc  : {val_acc:.4f}\n\n")
    f.write(report)

# Plot per-class precision / recall / F1 bar chart
from sklearn.metrics import precision_score, recall_score, f1_score
prec = precision_score(y_true, y_pred, average=None, zero_division=0)
rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
f1   = f1_score(y_true, y_pred, average=None, zero_division=0)

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, prec, width, label='Precision', color='#2196F3', alpha=0.85)
ax.bar(x,         rec,  width, label='Recall',    color='#4CAF50', alpha=0.85)
ax.bar(x + width, f1,   width, label='F1-Score',  color='#FF9800', alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Precision / Recall / F1', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)

plt.tight_layout()
plt.savefig(GRAPH_DIR / 'per_class_metrics.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Per-class metrics bar chart saved.")

# ── 11. ROC-AUC Curves ────────────────────────────────────────────────────────
print("\n[STEP 11] Generating ROC-AUC curves …")

y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
roc_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

fig, ax = plt.subplots(figsize=(10, 7))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:len(y_true), i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=roc_colors[i], lw=2,
            label=f'{cls}  (AUC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC-AUC Curves — One-vs-Rest', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / 'roc_auc_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ROC-AUC curves saved.")

# ── 12. Summary Graph (Dashboard) ────────────────────────────────────────────
print("\n[STEP 12] Generating summary dashboard …")

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# A) Combined accuracy
ax_acc = fig.add_subplot(gs[0, 0])
ax_acc.plot(x_p1, history1.history['accuracy'],          color='#2196F3', lw=1.5, label='Train P1')
ax_acc.plot(x_p1, history1.history['val_accuracy'],      color='#90CAF9', lw=1.5, label='Val P1')
ax_acc.plot(x_p2, history2.history['accuracy'],          color='#E91E63', lw=1.5, label='Train FT')
ax_acc.plot(x_p2, history2.history['val_accuracy'],      color='#F48FB1', lw=1.5, label='Val FT')
ax_acc.axvline(x=ep1, color='gray', linestyle='--', lw=1)
ax_acc.set_title('Accuracy', fontweight='bold'); ax_acc.legend(fontsize=7); ax_acc.grid(True, alpha=0.3)

# B) Combined loss
ax_loss = fig.add_subplot(gs[0, 1])
ax_loss.plot(x_p1, history1.history['loss'],             color='#2196F3', lw=1.5, label='Train P1')
ax_loss.plot(x_p1, history1.history['val_loss'],         color='#90CAF9', lw=1.5, label='Val P1')
ax_loss.plot(x_p2, history2.history['loss'],             color='#E91E63', lw=1.5, label='Train FT')
ax_loss.plot(x_p2, history2.history['val_loss'],         color='#F48FB1', lw=1.5, label='Val FT')
ax_loss.axvline(x=ep1, color='gray', linestyle='--', lw=1)
ax_loss.set_title('Loss', fontweight='bold'); ax_loss.legend(fontsize=7); ax_loss.grid(True, alpha=0.3)

# C) ROC-AUC
ax_roc = fig.add_subplot(gs[0, 2])
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:len(y_true), i])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=roc_colors[i], lw=1.5, label=f'{cls} ({roc_auc:.2f})')
ax_roc.plot([0,1],[0,1],'k--',lw=1)
ax_roc.set_title('ROC-AUC', fontweight='bold'); ax_roc.legend(fontsize=7); ax_roc.grid(True, alpha=0.3)

# D) Confusion matrix
ax_cm  = fig.add_subplot(gs[1, 0])
cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax_cm,
            linewidths=0.5, annot_kws={'size': 9})
ax_cm.set_title('Norm. Confusion Matrix', fontweight='bold')
ax_cm.tick_params(axis='x', rotation=20, labelsize=8)
ax_cm.tick_params(axis='y', rotation=0,  labelsize=8)

# E) Per-class F1
ax_f1  = fig.add_subplot(gs[1, 1])
ax_f1.bar(class_names, f1, color=roc_colors, alpha=0.85)
ax_f1.set_ylim(0, 1.1)
ax_f1.set_title('Per-Class F1 Score', fontweight='bold')
ax_f1.tick_params(axis='x', rotation=15, labelsize=9)
ax_f1.grid(axis='y', alpha=0.3)
for i, v in enumerate(f1):
    ax_f1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

# F) Class distribution
ax_dist = fig.add_subplot(gs[1, 2])
bar_x = np.arange(len(class_names))
ax_dist.bar(bar_x - 0.2, list(counts_before.values()), 0.35, label='Before', color='#9E9E9E', alpha=0.8)
ax_dist.bar(bar_x + 0.2, list(counts_after.values()),  0.35, label='After',  color='#4CAF50', alpha=0.8)
ax_dist.set_xticks(bar_x); ax_dist.set_xticklabels(class_names, rotation=15, fontsize=9)
ax_dist.set_title('Class Balancing', fontweight='bold')
ax_dist.legend(fontsize=9); ax_dist.grid(axis='y', alpha=0.3)

fig.suptitle('ResNet50 — ECG Classification Summary Dashboard', fontsize=16, fontweight='bold')
plt.savefig(GRAPH_DIR / 'summary_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Summary dashboard saved.")

# ── 13. Cleanup ───────────────────────────────────────────────────────────────
print("\n[STEP 13] Cleaning up balanced dataset …")
shutil.rmtree(BALANCED_DIR, ignore_errors=True)
print("  Temporary balanced data removed.")

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Best model : {best_model_path}")
print(f"  Graphs     : {GRAPH_DIR}/")
print(f"\n  Saved graphs:")
for g in sorted(GRAPH_DIR.iterdir()):
    print(f"    {g.name}")
print(f"\n  Final Val Accuracy : {val_acc:.4f}")
print(f"  Final Val Loss     : {val_loss:.4f}")
print("=" * 60)
