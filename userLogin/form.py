from django import forms
from captcha.fields import CaptchaField

class UserRegister(forms.Form):
    username = forms.CharField(label='注册用户名',max_length=10)
    password1 = forms.CharField(label='设置密码',widget=forms.PasswordInput())
    password2 = forms.CharField(label='确认密码',widget=forms.PasswordInput())
    email  =  forms.CharField(label='电子邮箱',max_length=30)
    phone =  forms.CharField(label='手机号码',max_length=11)
    type = forms.CharField(label='用户身份',max_length=20)

class UserLogin(forms.Form):
    username = forms.CharField(label='用户名',max_length=20)
    password = forms.CharField(label='密码',widget=forms.PasswordInput())

