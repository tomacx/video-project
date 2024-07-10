from django.db import models

# Create your models here.
class Face(models.Model):
    username = models.CharField(max_length=10)
    faceSave = models.ImageField(upload_to='D:/summerProject2024/video/faceRecog/images/')
