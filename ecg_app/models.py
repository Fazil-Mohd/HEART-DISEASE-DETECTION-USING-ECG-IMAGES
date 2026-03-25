from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patients')
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True, help_text="Optional: Email address to send reports to.")
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    medical_history = models.TextField(blank=True, help_text="Any known prior heart conditions, medications, etc.")
    contact_number = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.age}{self.gender})"
    
    class Meta:
        ordering = ['-created_at']

class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('user', 'Normal User'),
        ('clinic', 'Clinic / Organization'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile ({self.get_role_display()})"

class ECGRecord(models.Model):
    CATEGORY_CHOICES = [
        ('normal', 'Normal ECG'),
        ('abnormal', 'Abnormal Heartbeat'),
        ('mi', 'Myocardial Infarction'),
        ('post_mi', 'Post MI History'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ecg_records')
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True, blank=True, related_name='ecg_records', help_text="Optional: Link this ECG to a specific patient")
    image = models.ImageField(upload_to='uploaded_ecgs/%Y/%m/%d/')
    predicted_category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    confidence = models.FloatField(null=True, blank=True, default=None)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='completed')
    upload_date = models.DateTimeField(auto_now_add=True)
    processed_date = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)
    doctor_notes = models.TextField(blank=True)
    
    # Store prediction probabilities
    normal_prob = models.FloatField(default=0.0)
    abnormal_prob = models.FloatField(default=0.0)
    mi_prob = models.FloatField(default=0.0)
    post_mi_prob = models.FloatField(default=0.0)

    # LIME Explainability
    lime_images = models.JSONField(null=True, blank=True)  # {class_name: "relative/path/to/lime_class.png"}
    lime_explanation_data = models.JSONField(null=True, blank=True)  # {class_name: [(superpixel_id, weight), ...]}
    
    def __str__(self):
        return f"ECG #{self.id} - {self.get_predicted_category_display()}"
    
    class Meta:
        ordering = ['-upload_date']
        verbose_name = 'ECG Record'
        verbose_name_plural = 'ECG Records'

class TrainingSession(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    session_id = models.CharField(max_length=100, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=32)
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    training_time = models.FloatField(null=True, blank=True)  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"Training Session {self.session_id}"
    
    class Meta:
        ordering = ['-created_at']