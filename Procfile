web: gunicorn ecg_project.wsgi --workers 2 --timeout 120 --log-file -
worker: celery -A ecg_project worker -l info --pool=solo --concurrency=1
