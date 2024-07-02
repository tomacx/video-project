from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(primary_key=True,max_length=10)
    password = models.CharField(max_length=8)
