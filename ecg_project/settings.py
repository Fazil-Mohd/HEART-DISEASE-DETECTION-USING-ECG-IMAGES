import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-your-secret-key-here'
DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ecg_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ecg_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ecg_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ML Model Configuration
# ML Model Configuration
ML_CONFIG = {
    'MODEL_PATH': BASE_DIR / 'ecg_model.h5',
    'LABEL_ENCODER_PATH': BASE_DIR / 'label_encoder.pkl',
    'CLASS_NAMES_PATH': BASE_DIR / 'class_names.txt',
    'TRAINING_HISTORY_PATH': BASE_DIR / 'training_history.png',
    'DATASET_PATH': BASE_DIR / 'data',
    
    # Display names for the classes
    'CLASS_DISPLAY_NAMES': [
        'Normal ECG',
        'Abnormal Heartbeat', 
        'Myocardial Infarction',
        'Post MI History'
    ],
    
    # Mapping: folder names → class labels
    'FOLDER_TO_CLASS': {
        'normal_ecg_images': 'normal',
        'abnormal_heartbeat_ecg_images': 'abnormal',
        'myocardial_infarction_ecg_images': 'mi',
        'post_mi_history_ecg_images': 'post_mi'
    },
    
    # Folder names inside your data directory
    'DATASET_FOLDERS': [
        'normal_ecg_images',
        'abnormal_heartbeat_ecg_images', 
        'myocardial_infarction_ecg_images',
        'post_mi_history_ecg_images'
    ],
    
    # Class labels (used for prediction)
    'CLASS_LABELS': ['normal', 'abnormal', 'mi', 'post_mi'],
}

# Authentication URLs
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'home'

# Celery Configuration
CELERY_BROKER_URL = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE