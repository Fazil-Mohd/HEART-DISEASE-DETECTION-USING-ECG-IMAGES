# urls.py - User-focused version
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Authentication URLs
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Email Verification URLs
    path('verify-email/<uuid:token>/', views.verify_email_view, name='verify_email'),
    path('verify-email/pending/', views.email_verification_pending_view, name='email_pending'),
    path('resend-verification/', views.resend_verification_view, name='resend_verification'),

    
    # Core Application URLs (User actions only)
    path('', views.home_view, name='home'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('upload/', views.upload_ecg_view, name='upload'),
    path('result/<int:ecg_id>/', views.ecg_result_view, name='ecg_result'),
    path('history/', views.ecg_history_view, name='history'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin-login/', views.admin_login_view, name='admin_login'),

    # In urls.py, add this to urlpatterns:
    path('export-csv/', views.export_history_csv_view, name='export_history_csv_view'),
    path('model-performance/', views.model_performance_view, name='model_performance'),

    # Patient Management URLs
    path('patients/', views.patient_list_view, name='patient_list'),
    path('patients/new/', views.patient_create_view, name='patient_create'),
    path('patients/<int:patient_id>/', views.patient_detail_view, name='patient_detail'),
    path('patients/<int:patient_id>/edit/', views.patient_update_view, name='patient_update'),
    path('patients/<int:patient_id>/delete/', views.patient_delete_view, name='patient_delete'),

    # API URLs (User actions only)
    path('api/train/', views.api_train_model, name='api_train'),
    path('api/user-stats/', views.api_user_stats, name='api_user_stats'),
    path('api/lime/<int:ecg_id>/', views.api_lime_explanation, name='api_lime_explanation'),
    path('pdf/<int:ecg_id>/', views.generate_pdf_report, name='generate_pdf_report'),

    # Password management (keep these for user convenience)
    path('password-reset/', 
         auth_views.PasswordResetView.as_view(
             template_name='ecg_app/auth/password_reset.html',
             email_template_name='ecg_app/auth/password_reset_email.html',
             subject_template_name='ecg_app/auth/password_reset_subject.txt'
         ), 
         name='password_reset'),
    
    path('password-reset/done/', 
         auth_views.PasswordResetDoneView.as_view(
             template_name='ecg_app/auth/password_reset_done.html'
         ), 
         name='password_reset_done'),
    
    path('password-reset-confirm/<uidb64>/<token>/', 
         auth_views.PasswordResetConfirmView.as_view(
             template_name='ecg_app/auth/password_reset_confirm.html'
         ), 
         name='password_reset_confirm'),
    
    path('password-reset-complete/', 
         auth_views.PasswordResetCompleteView.as_view(
             template_name='ecg_app/auth/password_reset_complete.html'
         ), 
         name='password_reset_complete'),
    
    # For changing password while logged in
    path('password-change/', 
         auth_views.PasswordChangeView.as_view(
             template_name='ecg_app/auth/password_change.html'
         ), 
         name='password_change'),
    
    path('password-change/done/', 
         auth_views.PasswordChangeDoneView.as_view(
             template_name='ecg_app/auth/password_change_done.html'
         ), 
         name='password_change_done'),
]