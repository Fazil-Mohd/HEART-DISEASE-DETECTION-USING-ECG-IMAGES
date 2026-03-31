# ecg_app/admin.py
from django.contrib import admin
from .models import ECGRecord, UserProfile, EmailVerificationToken

admin.site.register(ECGRecord)
admin.site.register(UserProfile)


@admin.register(EmailVerificationToken)
class EmailVerificationTokenAdmin(admin.ModelAdmin):
    list_display  = ('user', 'token', 'created_at', 'is_verified')
    list_filter   = ('is_verified',)
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('token', 'created_at')