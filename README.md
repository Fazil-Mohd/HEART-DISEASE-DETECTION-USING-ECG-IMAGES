# Automated Heart Disease Detection Using ECG Image Analysis

## Overview
This is a Django-based web application that provides automated heart disease detection by analyzing Electrocardiogram (ECG) images. It utilizes a deep learning model (ResNet) to classify ECG images into different cardiac conditions and provides explainability using LIME (Local Interpretable Model-agnostic Explanations).

## Features
- **ECG Image Upload and Analysis:** Users can upload scanned ECG images to receive instant predictions.
- **Deep Learning Classification:** Uses a trained ResNet model (`resnet_models`) for accurate detection.
- **Explainable AI (LIME):** Generates visual explanations highlighting the regions of the ECG that influenced the model's prediction.
- **Role-Based Access Control:** Separate roles for "Normal Users" (managing their own scans) and "Clinics/Organizations" (managing multiple patients).
- **Patient Management:** Clinics can create patient profiles and track their ECG history over time.
- **Diagnosis Trends:** Compares past and current ECG scans to analyze disease progression.
- **PDF Reporting:** Automatically generates downloadable PDF reports for each ECG analysis.
- **Task Queue Integration:** Uses Celery and Redis to handle model inference and email/reporting tasks asynchronously.

## Technologies Used
- **Backend:** Django 5.2, Python
- **Frontend:** HTML, CSS, JavaScript (Vanilla/Bootstrap)
- **Machine Learning:** TensorFlow, Keras, Scikit-learn, Scikit-image
- **Explainability:** LIME
- **Task Queue:** Celery, Redis
- **Database:** SQLite (default)
- **Image Processing & PDF:** Pillow, OpenCV, xhtml2pdf, Matplotlib

## Installation & Setup

### Prerequisites
- Python 3.9+
- Redis Server (for Celery)

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Automated-Heart-Disease-Detection-Using-ECG-Image-Analysis"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply database migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start the Redis server** (Make sure Redis is running on `localhost:6379`).

6. **Start the Celery worker** (in a new terminal, with the virtual environment activated):
   ```bash
   celery -A ecg_project worker -l info --pool=solo
   ```

7. **Run the Django development server:**
   ```bash
   python manage.py runserver
   ```

8. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:8000`.

## Project Structure
- `ecg_app/`: Main Django app containing views, models, and tasks.
- `ecg_project/`: Django project configuration (settings, urls).
- `data/` & `resnet_models/`: Contains the datasets and pre-trained deep learning models.
- `media/`: Stores user-uploaded ECG images and generated LIME explanations.
- `static/`: Static files (CSS, JS, images).
- `templates/`: HTML templates for the frontend.

## License
MIT License
