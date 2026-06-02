web: gunicorn ecg_project.wsgi --workers 1 --timeout 300 --log-file -
worker: celery -A ecg_project worker -l info --pool=solo --concurrency=1

