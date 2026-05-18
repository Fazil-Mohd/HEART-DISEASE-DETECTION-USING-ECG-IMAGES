#!/usr/bin/env bash
# build.sh — Render build script
# Render runs this script every time you push a new deployment.
# Set this as the "Build Command" in your Render web service settings.

set -e  # exit immediately on any error

echo "==> Installing Python dependencies …"
pip install -r requirements.txt

echo "==> Collecting static files …"
python manage.py collectstatic --noinput

echo "==> Running database migrations …"
python manage.py migrate --noinput

echo "==> Build complete ✅"
