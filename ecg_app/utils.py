import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import joblib
from django.conf import settings

class ECGClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.class_names = []
        self.load_model()
        
    def load_model(self):
        """Load the trained model and artifacts"""
        try:
            # Load model
            self.model = keras.models.load_model(str(settings.ML_CONFIG['MODEL_PATH']))
            
            # Load label encoder
            self.label_encoder = joblib.load(str(settings.ML_CONFIG['LABEL_ENCODER_PATH']))
            self.class_names = list(self.label_encoder.classes_)
            
            print(f"ECG Classifier loaded successfully.")
            print(f"Classes: {self.class_names}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # If image is file path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image from {image}")
        else:
            # Assume image is numpy array or file object
            img = image
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (224, 224))
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image):
        """Make prediction on ECG image"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model could not be loaded")
        
        # Preprocess image
        preprocessed_img = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(preprocessed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        # Get class name
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get all probabilities
        all_probs = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = self.label_encoder.inverse_transform([idx])[0]
            all_probs[class_name] = float(prob)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'predicted_class_idx': predicted_class_idx
        }
    
    def batch_predict(self, images):
        """Make predictions on multiple images"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model could not be loaded")
        
        # Preprocess all images
        preprocessed_images = []
        for image in images:
            preprocessed_img = self.preprocess_image(image)
            preprocessed_images.append(preprocessed_img[0])  # Remove batch dimension
        
        if not preprocessed_images:
            return []
        
        X = np.array(preprocessed_images)
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        results = []
        for pred in predictions:
            predicted_class_idx = np.argmax(pred)
            confidence = float(pred[predicted_class_idx])
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            all_probs = {}
            for idx, prob in enumerate(pred):
                class_name = self.label_encoder.inverse_transform([idx])[0]
                all_probs[class_name] = float(prob)
            
            results.append({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        
        return results

# Singleton instance
ecg_classifier = ECGClassifier()