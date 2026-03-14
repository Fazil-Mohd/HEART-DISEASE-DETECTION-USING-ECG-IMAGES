# ecg_app/admin.py
from django.contrib import admin
from .models import ECGRecord, UserProfile

admin.site.register(ECGRecord)
admin.site.register(UserProfile)