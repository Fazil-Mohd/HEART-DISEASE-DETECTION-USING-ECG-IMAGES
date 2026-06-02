import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── SECURITY ───────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-local-dev-key-change-in-production'
)

# Read DEBUG from env; defaults to True locally so dev experience is unchanged.
DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'

# In production set ALLOWED_HOSTS env var to: your-app.onrender.com
_raw_hosts = os.environ.get('ALLOWED_HOSTS', '')
ALLOWED_HOSTS = [h.strip() for h in _raw_hosts.split(',') if h.strip()] or ['*']

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
    # whitenoise must come directly after SecurityMiddleware
    'whitenoise.middleware.WhiteNoiseMiddleware',
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
# Locally: SQLite (DATABASE_URL not set → falls back to sqlite3)
# Production (Render): DATABASE_URL is set automatically by Render PostgreSQL
_database_url = os.environ.get('DATABASE_URL', '')
if _database_url:
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.config(
            default=_database_url,
            conn_max_age=600,
            ssl_require=True,
        )
    }
else:
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

# whitenoise: serve compressed static files efficiently in production
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

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
        'normal_ecg_images':                'normal',
        'abnormal_heartbeat_ecg_images':    'abnormal',
        'myocardial_infarction_ecg_images': 'mi',
        'post_mi_history_ecg_images':       'post_mi',
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
# Locally: defaults to localhost Redis.
# Production (Render): REDIS_URL env var is set automatically by Render Redis.
_redis_url = os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/0')
CELERY_BROKER_URL     = _redis_url
CELERY_RESULT_BACKEND = _redis_url
CELERY_ACCEPT_CONTENT    = ['json']
CELERY_TASK_SERIALIZER   = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# Acknowledge tasks only AFTER they finish — prevents task loss if Celery crashes mid-run
CELERY_TASK_ACKS_LATE = True
# Process one task at a time per worker thread — avoids memory pressure with TensorFlow
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# CRITICAL: fail fast when Redis is unavailable — prevents 60-second retry loop
# that would cause gunicorn worker timeout during registration.
CELERY_BROKER_CONNECTION_RETRY    = False   # don't retry on startup
CELERY_BROKER_CONNECTION_MAX_RETRIES = 1    # at most 1 retry then give up
CELERY_BROKER_TRANSPORT_OPTIONS   = {
    'socket_timeout':         3,   # seconds to wait for response
    'socket_connect_timeout': 3,   # seconds to wait for connection
}

# ── EMAIL ──────────────────────────────────────────────────────────────────────
# All credentials are read from environment variables — never hard-coded.
#
# HOW TO SET UP GMAIL:
#   1. Enable 2-Step Verification on your Google account
#      https://myaccount.google.com/security
#   2. Generate an App Password (16 chars, no spaces):
#      Google Account → Security → App passwords → Mail + Windows Computer
#   3. Set env vars:
#      Local  : add to your .env file (see .env.example)
#      Render : add via the Render dashboard → Environment
#
EMAIL_BACKEND       = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST          = 'smtp.gmail.com'
EMAIL_PORT          = 587
EMAIL_USE_TLS       = True
EMAIL_HOST_USER     = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL  = os.environ.get('EMAIL_HOST_USER', '')
# Timeout for SMTP connection — prevents gunicorn worker timeout (120s)
# when Gmail is slow or credentials are wrong.
EMAIL_TIMEOUT       = 10  # seconds

# ── PRODUCTION SECURITY HEADERS (only when DEBUG=False) ───────────────────────
if not DEBUG:
    SECURE_SSL_REDIRECT              = True   # redirect all HTTP → HTTPS
    SECURE_BROWSER_XSS_FILTER        = True
    SECURE_CONTENT_TYPE_NOSNIFF      = True
    X_FRAME_OPTIONS                   = 'DENY'
    SECURE_HSTS_SECONDS              = 31536000   # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS   = True
    SECURE_PROXY_SSL_HEADER          = ('HTTP_X_FORWARDED_PROTO', 'https')
    SESSION_COOKIE_SECURE            = True
    CSRF_COOKIE_SECURE               = True