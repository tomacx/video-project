import pickle
from urllib import request
import json

from django.http import HttpResponse
from django.shortcuts import render, redirect

import modify
from userLogin.models import User
import hashlib
from django.template import RequestContext
from userLogin.form import UserLogin,UserRegister
from modify import models

# Create your views here.
def take_md5(content):
    content_bytes = content.encode('utf-8') #将字符串转为字节对象
    hash = hashlib.md5()    #创建hash加密实例
    hash.update(content_bytes)    #hash加密
    result = hash.hexdigest()  #得到加密结果
    return result[:16]

def login(request):
    if request.method == 'POST':
        form = UserLogin(request.POST)
        if form.is_valid():
            username = request.POST.get('username')
            password = request.POST.get('password')
            password = take_md5(password)
            namefilter = User.objects.filter(username=username,password=password)
            print(username)
            if len(namefilter) > 0:
                return redirect("/index")
            else:
                return render(request,'login.html',{'form':form})
        else:
            form = UserLogin()
            return render(request,'login.html',{'form':form})
    else:
        return render(request,'login.html')

def register(request):
    if request.method == 'POST':
        form = UserRegister(request.POST)
        print(form.errors)
        if form.is_valid():
            username = form.cleaned_data['username']
            namefilter = User.objects.filter(username = username)
            if namefilter.exists():
                return HttpResponse(200)
            else:
                password1 = form.cleaned_data['password1']
                password2 = form.cleaned_data['password2']
                if password1 != password2:
                    return render('register',{'error':'两次输入的密码不一致'})
                else:
                    password = take_md5(password1)
                    email = form.cleaned_data['email']
                    phone = form.cleaned_data['phone']
                    type = form.cleaned_data['type']
                    # 将表单写入数据库
                    user = User.objects.create(username=username, password=password, email=email,
                                                   phone=phone,type=type)
                    user1 = modify.models.User.objects.create(username=username, password=password,email=email,phone=phone,type=type)
                    user.save()
                    user1.save()
                    return render(request,'login.html')
        form = UserRegister()
        return render(request,'register.html')
    else:
        return render(request,'register.html')

def index(request):
    return render(request,'index.html')