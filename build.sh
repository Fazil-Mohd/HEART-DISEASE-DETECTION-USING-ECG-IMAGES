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

echo "==> Creating superuser if not exists …"
python manage.py shell -c "
import os
from django.contrib.auth.models import User
u = os.environ.get('DJANGO_SUPERUSER_USERNAME','')
e = os.environ.get('DJANGO_SUPERUSER_EMAIL','')
p = os.environ.get('DJANGO_SUPERUSER_PASSWORD','')
if u and p and not User.objects.filter(username=u).exists():
    User.objects.create_superuser(u, e, p)
    print(f'Superuser {u!r} created.')
else:
    print('Superuser already exists or env vars not set — skipped.')
"

echo "==> Build complete ✅"
