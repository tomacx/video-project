# Generated by Django 5.0.4 on 2024-07-03 14:23

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('username', models.CharField(max_length=10, primary_key=True, serialize=False, verbose_name='用户名')),
                ('password', models.CharField(max_length=16, verbose_name='密码')),
                ('phone', models.CharField(default=1, max_length=11, verbose_name='电话号码')),
                ('email', models.CharField(default='1@qq.com', max_length=30, verbose_name='注册邮箱')),
                ('type', models.CharField(default='场务人员', max_length=20, verbose_name='身份')),
            ],
        ),
    ]
