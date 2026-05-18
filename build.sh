#!/usr/bin/env bash
set -e

echo "==> Installing Python dependencies …"
pip install -r requirements.txt

echo "==> Downloading ML model from Hugging Face Hub …"
python download_model.py

echo "==> Collecting static files …"
python manage.py collectstatic --noinput

echo "==> Running database migrations …"
python manage.py migrate --noinput

echo "==> Build complete ✅"
