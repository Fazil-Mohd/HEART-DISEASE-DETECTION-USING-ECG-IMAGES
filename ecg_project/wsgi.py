"""
WSGI config for ecg_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ecg_project.settings')

# ── Auto-download model from Hugging Face Hub if not present ──────────────────
# This runs once at server startup. In local dev the file already exists so
# this is a fast no-op. On Render it downloads the model before serving traffic.
try:
    from download_model import download_model_if_needed
    download_model_if_needed()
except Exception as _dl_err:
    import logging
    logging.getLogger(__name__).warning(
        f"Model download skipped: {_dl_err}"
    )

application = get_wsgi_application()

