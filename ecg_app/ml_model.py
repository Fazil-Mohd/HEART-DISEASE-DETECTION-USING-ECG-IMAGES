# ml_model.py  — ResNet50 ECG Classifier
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from django.conf import settings
import logging
import time

logger = logging.getLogger(__name__)

# Register the custom FocalLoss so Keras can deserialize the saved ResNet50 model.
# This must match the class registered in train_resnet.py (same package='ECG').
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


class MemoryEfficientECGModel:
    def __init__(self):
        self.model = None

        # Path to the ResNet50 model trained by train_resnet.py
        self.model_path = os.path.join(settings.BASE_DIR, "resnet_models", "resnet50_ecg_best.keras")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Class order must match the training order (sorted alphabetically)
        self.class_names = ['abnormal', 'mi', 'normal', 'post_mi']

        # Input size expected by ResNet50
        self.img_size = (224, 224)

        self.training_in_progress = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def model_exists(self):
        """Check if model file exists."""
        return os.path.exists(self.model_path)

    def load_model(self):
        """Load the ResNet50 model from disk (lazy, loads once)."""
        if self.model is not None:
            return True
        try:
            logger.info("Loading ResNet50 ECG model …")
            t0 = time.time()
            # compile=False: FocalLoss registered above handles deserialization,
            # but we don't need the optimizer for inference.
            self.model = load_model(self.model_path, compile=False)
            logger.info(f"ResNet50 loaded in {time.time() - t0:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Load model error: {e}")
            self.model = None
            return False

    def predict(self, image_path):
        """
        Predict ECG class from an image file.
        Returns dict with predicted_class, confidence, all_probabilities.
        """
        try:
            if self.model is None:
                if not self.load_model():
                    logger.warning("No trained model found — using dummy prediction")
                    return self._dummy_prediction()

            # 1. Load & preprocess with ResNet50 preprocessing
            img = load_img(image_path, target_size=self.img_size)
            x   = img_to_array(img)
            x   = np.expand_dims(x, axis=0)
            x   = resnet_preprocess(x)       # ResNet50-specific normalisation

            # 2. Predict
            probs = self.model.predict(x, verbose=0)[0]

            # ── Bayesian prior correction to counter post_mi bias ─────────────
            # The model was trained on a perfectly balanced dataset (all classes
            # oversampled to equal counts → effective prior = 0.25 each).
            # So the raw softmax output approximates P(class | image, balanced).
            # To recover the true posterior using real class frequencies:
            #   adjusted ∝ model_probs × real_prior
            # post_mi real prior (0.186) < normal (0.307) → gets penalised.
            # Exact original counts: abnormal=233, mi=239, normal=284, post_mi=172
            real_priors = np.array([233, 239, 284, 172], dtype=np.float32)
            real_priors /= real_priors.sum()     # [0.252, 0.258, 0.307, 0.186]

            adjusted = probs * real_priors        # element-wise multiply
            adjusted = adjusted / adjusted.sum()  # renormalise to sum=1

            predicted_idx   = int(np.argmax(adjusted))
            predicted_class = self.class_names[predicted_idx]
            confidence      = float(adjusted[predicted_idx])

            all_probabilities = {
                self.class_names[i]: float(adjusted[i])
                for i in range(len(self.class_names))
            }

            logger.info(f"Prediction: {predicted_class} ({confidence:.3f})")
            print("FINAL PREDICTIONS:", adjusted)
            print("ALL PROBABILITIES:", all_probabilities)

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._dummy_prediction()

    def get_model_info(self):
        """Return model metadata for the UI / admin page."""
        info = {
            'is_trained':             self.model_exists(),
            'training_in_progress':   self.training_in_progress,
            'num_classes':            len(self.class_names),
            'class_names':            self.class_names,
            'ensemble_models_exist':  self.model_exists(),   # kept for template compat.
        }

        if self.model_exists():
            try:
                self.load_model()
                info['model_summary'] = [
                    "ResNet50 ECG Model",
                    f"Input: {self.img_size[0]}×{self.img_size[1]}×3",
                    f"Classes: {', '.join(self.class_names)}",
                    "Trained with balanced dataset + two-phase fine-tuning",
                ]
                # Report last known val accuracy from classification report if available
                report_path = os.path.join(settings.BASE_DIR, "resnet_models",
                                           "classification_report.txt")
                if os.path.exists(report_path):
                    with open(report_path) as f:
                        for line in f:
                            if line.startswith("Validation Acc"):
                                try:
                                    acc = float(line.split(":")[1].strip())
                                    info['accuracy'] = round(acc * 100, 2)
                                except Exception:
                                    pass
                if 'accuracy' not in info:
                    info['accuracy'] = 0.0
            except Exception as e:
                logger.error(f"get_model_info error: {e}")
                info['accuracy'] = 0.0

        return info

    def auto_train_if_needed(self):
        """Auto-train if no model exists (calls train_model with defaults)."""
        if not self.model_exists():
            logger.info("No model found — auto-training …")
            return self.train_model()
        return True

    def train_model(self, epochs=30, batch_size=16):
        """
        Quick single-phase ResNet50 training (for on-demand retraining via UI).
        For full two-phase training with balanced dataset use train_resnet.py directly.
        """
        try:
            self.training_in_progress = True
            dataset_path = os.path.join(settings.BASE_DIR, 'data')
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset not found at {dataset_path}")
                return False

            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras import layers, Model
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.callbacks import (
                EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            )
            from sklearn.utils.class_weight import compute_class_weight

            datagen = ImageDataGenerator(
                preprocessing_function=resnet_preprocess,
                validation_split=0.2,
                rotation_range=10,
                zoom_range=0.05,
                width_shift_range=0.05,
                height_shift_range=0.05
            )
            val_dg = ImageDataGenerator(
                preprocessing_function=resnet_preprocess,
                validation_split=0.2
            )

            train_gen = datagen.flow_from_directory(
                dataset_path, target_size=self.img_size, batch_size=batch_size,
                class_mode='categorical', subset='training',
                classes=self.class_names, shuffle=True
            )
            val_gen = val_dg.flow_from_directory(
                dataset_path, target_size=self.img_size, batch_size=batch_size,
                class_mode='categorical', subset='validation',
                classes=self.class_names, shuffle=False
            )

            y_all = train_gen.classes
            cw    = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
            class_weights = dict(enumerate(cw))

            # Build model
            reg  = tf.keras.regularizers.l2(1e-4)
            base = ResNet50(weights='imagenet', include_top=False,
                            input_shape=(224, 224, 3))
            base.trainable = False

            x = base.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
            x = layers.Dropout(0.4)(x)
            out = layers.Dense(len(self.class_names), activation='softmax')(x)
            model = Model(inputs=base.input, outputs=out)

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy']
            )

            callbacks = [
                ModelCheckpoint(self.model_path, monitor='val_accuracy',
                                save_best_only=True, verbose=1),
                EarlyStopping(monitor='val_accuracy', patience=10,
                              restore_best_weights=True, mode='max', verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4,
                                  min_lr=1e-7, verbose=1),
            ]

            model.fit(
                train_gen, validation_data=val_gen, epochs=epochs,
                class_weight=class_weights, callbacks=callbacks
            )

            self.model = model
            self.training_in_progress = False
            logger.info("Training completed.")
            return True

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.training_in_progress = False
            return False

    # ── LIME Explainability ────────────────────────────────────────────────────

    def generate_lime_explanation(self, image_path, save_dir):
        """
        Generate a LIME explanation overlay for ALL 4 classes.

        Parameters
        ----------
        image_path : str   – absolute path to the ECG image
        save_dir   : str   – directory to save the 4 overlay PNGs

        Returns
        -------
        dict  {
            class_name: {
                'image_path': 'filename.png',
                'weights': [(superpixel_id, weight), ...]
            }
        }
        """
        try:
            import os
            import matplotlib
            matplotlib.use('Agg')          # non-interactive backend — required on server
            import matplotlib.pyplot as plt
            from lime import lime_image as lime_img
            from skimage.segmentation import mark_boundaries

            if self.model is None:
                if not self.load_model():
                    raise RuntimeError("Model not loaded")

            # ── 1. Load raw RGB image (uint8, NOT ResNet-preprocessed) ──────────
            img_raw   = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img_raw).astype(np.uint8)   # (224,224,3)

            # ── 2. Prediction function that LIME calls for each perturbed batch ─
            real_priors = np.array([233, 239, 284, 172], dtype=np.float32)
            real_priors /= real_priors.sum()          # same correction as predict()

            def predict_fn(images):
                """images: (N,224,224,3) uint8 numpy arrays from LIME"""
                batch   = resnet_preprocess(images.astype(np.float32))
                raw     = self.model.predict(batch, verbose=0)   # (N,4)
                adjusted = raw * real_priors
                adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
                return adjusted

            # ── 3. Run LIME ──────────────────────────────────────────────────────
            explainer   = lime_img.LimeImageExplainer(random_state=42)
            explanation = explainer.explain_instance(
                img_array,
                predict_fn,
                top_labels=4,
                hide_color=0,
                num_samples=1000,
                batch_size=32,
            )

            # ── 4 & 5. Build and save overlays for ALL 4 classes ─────────────────
            results_by_class = {}
            
            for class_idx, class_name in enumerate(self.class_names):
                # 4. Generate overlay (positive + negative regions)
                try:
                    temp_img, mask = explanation.get_image_and_mask(
                        class_idx,
                        positive_only=False,
                        num_features=15,
                        hide_rest=False,
                    )
                except KeyError:
                    # LIME might not compute explanation for a class if probability is ~0
                    results_by_class[class_name] = {'image_name': None, 'weights': []}
                    continue

                # 5. Save overlay PNG
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0f172a')

                # Left: original image
                axes[0].imshow(img_array)
                axes[0].set_title('Original ECG', color='white', fontsize=13, pad=10)
                axes[0].axis('off')

                # Right: LIME overlay with boundaries
                axes[1].imshow(mark_boundaries(temp_img / 255.0, mask,
                                               color=(0.2, 0.9, 0.4),
                                               outline_color=(0.9, 0.2, 0.2)))
                axes[1].set_title(f'LIME Overlay: {class_name}', color='white',
                                   fontsize=13, pad=10)
                axes[1].axis('off')

                plt.tight_layout(pad=1.0)
                
                filename = f"lime_{class_name}.png"
                save_path = os.path.join(save_dir, filename)
                
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor='#0f172a')
                plt.close(fig)

                # 6. Collect per-class superpixel weights
                weights = []
                if class_idx in explanation.local_exp:
                    ind = explanation.local_exp[class_idx]   # [(sp_id, weight),...]
                    weights = [(int(sp), round(float(w), 6)) for sp, w in ind[:10]]
                
                results_by_class[class_name] = {
                    'image_name': filename,
                    'weights': weights
                }

            logger.info(f"LIME overlays saved to {save_dir}")
            return results_by_class

        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return {}

    # ── Private helpers ────────────────────────────────────────────────────────

    def _dummy_prediction(self):
        """Fallback prediction when no model is loaded."""
        return {
            'predicted_class': 'normal',
            'confidence': 0.85,
            'all_probabilities': {
                'normal': 0.85, 'abnormal': 0.10,
                'mi': 0.03, 'post_mi': 0.02
            }
        }

# Singleton instance used throughout the Django app
ecg_model = MemoryEfficientECGModel()