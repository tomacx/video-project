from django.db import models


# Create your models here.
class User(models.Model):
    username = models.CharField(primary_key=True,max_length=10,verbose_name='用户名')
    password = models.CharField(max_length=16,verbose_name='密码')
    phone = models.CharField(max_length=11,verbose_name='电话号码',default=1)
    email = models.CharField(max_length=30,verbose_name='注册邮箱',default='1@qq.com')
    type = models.CharField(max_length=20, verbose_name='身份',default='场务人员')

