import pickle
from urllib import request
import json
from django.shortcuts import render
from userLogin.models import User
import hashlib
from django.template import RequestContext
from userLogin.form import UserLogin,UserRegister


# Create your views here.
def take_md5(content):
    hash = hashlib.md5()    #创建hash加密实例
    hash.update(content)    #hash加密
    result = hash.hexdigest()  #得到加密结果
    return result

def login(request):
    if request.method == 'POST':
        form = UserLogin(request.POST)
        if form.is_valid():
            username = request.POST.get('username')
            password = request.POST.get('password')
            password = take_md5(password)
            namefilter = User.objects.filter(username=username,password=password)
            if len(namefilter) > 0:
                return render('success.html',{'username':username,'operation':'登录'})
            else:
                return render('login.html', {'error': '该用户名不存在！'})
        else:
            form = UserLogin()
            return render(request,'login.html', {'form':form})


def register(request):
    if request.method == 'POST':
        form = UserRegister(request.POST)
        print(form.errors)
        if form.is_valid():
            username = form.cleaned_data['username']
            namefilter = User.objects.filter(username = username)
            if namefilter.exists():
                return render('register.html',{'error':'用户名已存在'})
            else:
                password1 = form.cleaned_data['password1']
                password2 = form.cleaned_data['password2']
                if password1 != password2:
                    return render('register.html',{'error':'两次输入的密码不一致'})
                else:
                    password = take_md5(password1)
                    email = form.cleaned_data['email']
                    phone = form.cleaned_data['phone']
                    type = form.cleaned_data['type']
                    # 将表单写入数据库
                    user = User.objects.create(username=username, password=password, email=email,
                                                   phone=phone,type=type)
                    user.save()
                    return render('success.html',{'username':username,'operation':'注册'})
    else:
        form = UserRegister()
        return render(request,'register.html',{'form',form})

