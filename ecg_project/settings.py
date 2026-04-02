import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── SECURITY ───────────────────────────────────────────────────────────────────
# IMPORTANT: Move SECRET_KEY to an environment variable before deploying.
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-your-secret-key-here')
DEBUG = True
ALLOWED_HOSTS = ['*']

# ── APPS ───────────────────────────────────────────────────────────────────────
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

# ── DATABASE ───────────────────────────────────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ── PASSWORD VALIDATION ────────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ── INTERNATIONALISATION ───────────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ── STATIC & MEDIA ─────────────────────────────────────────────────────────────
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── AUTHENTICATION ─────────────────────────────────────────────────────────────
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'home'

# ── CELERY ─────────────────────────────────────────────────────────────────────
CELERY_BROKER_URL    = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'
CELERY_ACCEPT_CONTENT   = ['json']
CELERY_TASK_SERIALIZER  = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# Acknowledge tasks only AFTER they finish — prevents task loss if Celery crashes mid-run
CELERY_TASK_ACKS_LATE = True
# Process one task at a time per worker thread — avoids memory pressure with TensorFlow
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# ── EMAIL ──────────────────────────────────────────────────────────────────────
#
# HOW TO SET UP GMAIL:
#   1. Enable 2-Step Verification on your Google account
#      https://myaccount.google.com/security
#
#   2. Generate an App Password (16 characters, no spaces):
#      Google Account → Security → App passwords
#      Select "Mail" + "Windows Computer" → Generate
#
#   3. Set these two environment variables on your machine:
#      Windows (Command Prompt):
#        setx EMAIL_HOST_USER "yourname@gmail.com"
#        setx EMAIL_HOST_PASSWORD "abcdefghijklmnop"
#      (Restart your terminal / Celery worker after setting them)
#
#   OR for quick local testing, replace os.environ.get(...) with your
#   actual values directly — but never commit real credentials to git.
#
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── SECURITY ───────────────────────────────────────────────────────────────────
# IMPORTANT: Move SECRET_KEY to an environment variable before deploying.
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-your-secret-key-here')
DEBUG = True
ALLOWED_HOSTS = ['*']

# ── APPS ───────────────────────────────────────────────────────────────────────
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

# ── DATABASE ───────────────────────────────────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ── PASSWORD VALIDATION ────────────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ── INTERNATIONALISATION ───────────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ── STATIC & MEDIA ─────────────────────────────────────────────────────────────
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── ML MODEL CONFIGURATION ─────────────────────────────────────────────────────
ML_CONFIG = {
    'MODEL_PATH': BASE_DIR / 'ecg_model.h5',
    'LABEL_ENCODER_PATH': BASE_DIR / 'label_encoder.pkl',
    'CLASS_NAMES_PATH': BASE_DIR / 'class_names.txt',
    'TRAINING_HISTORY_PATH': BASE_DIR / 'training_history.png',
    'DATASET_PATH': BASE_DIR / 'data',

    'CLASS_DISPLAY_NAMES': [
        'Normal ECG',
        'Abnormal Heartbeat',
        'Myocardial Infarction',
        'Post MI History',
    ],

    'FOLDER_TO_CLASS': {
        'normal_ecg_images':              'normal',
        'abnormal_heartbeat_ecg_images':  'abnormal',
        'myocardial_infarction_ecg_images': 'mi',
        'post_mi_history_ecg_images':     'post_mi',
    },

    'DATASET_FOLDERS': [
        'normal_ecg_images',
        'abnormal_heartbeat_ecg_images',
        'myocardial_infarction_ecg_images',
        'post_mi_history_ecg_images',
    ],

    'CLASS_LABELS': ['normal', 'abnormal', 'mi', 'post_mi'],
}

# ── AUTHENTICATION ─────────────────────────────────────────────────────────────
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'home'

# ── CELERY ─────────────────────────────────────────────────────────────────────
CELERY_BROKER_URL    = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'
CELERY_ACCEPT_CONTENT   = ['json']
CELERY_TASK_SERIALIZER  = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# Acknowledge tasks only AFTER they finish — prevents task loss if Celery crashes mid-run
CELERY_TASK_ACKS_LATE = True
# Process one task at a time per worker thread — avoids memory pressure with TensorFlow
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# ── EMAIL ──────────────────────────────────────────────────────────────────────
#
# HOW TO SET UP GMAIL:
#   1. Enable 2-Step Verification on your Google account
#      https://myaccount.google.com/security
#
#   2. Generate an App Password (16 characters, no spaces):
#      Google Account → Security → App passwords
#      Select "Mail" + "Windows Computer" → Generate
#
#   3. Set these two environment variables on your machine:
#      Windows (Command Prompt):
#        setx EMAIL_HOST_USER "yourname@gmail.com"
#        setx EMAIL_HOST_PASSWORD "abcdefghijklmnop"
#      (Restart your terminal / Celery worker after setting them)
#
#   OR for quick local testing, replace os.environ.get(...) with your
#   actual values directly — but never commit real credentials to git.
#
EMAIL_BACKEND       = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST          = 'smtp.gmail.com'
EMAIL_PORT          = 587
EMAIL_USE_TLS       = True
EMAIL_HOST_USER     = 'fazilhp11@gmail.com'
EMAIL_HOST_PASSWORD = 'tmpuwpyjyevswjbq'   # no spaces
DEFAULT_FROM_EMAIL  = 'fazilhp11@gmail.com'
