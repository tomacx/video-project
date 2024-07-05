import hashlib

from django.shortcuts import render, redirect

import modify
import userLogin
from modify.models import User
from userLogin.models import User

# Create your views here.
from django.shortcuts import render,HttpResponse,redirect
def take_md5(content):
    content_bytes = content.encode('utf-8') #将字符串转为字节对象
    hash = hashlib.md5()    #创建hash加密实例
    hash.update(content_bytes)    #hash加密
    result = hash.hexdigest()  #得到加密结果
    return result[:16]
def userinfo(request):
    username = request.POST.get('username', '')
    print(username)
    if len(username) == 0:
        user_list = modify.models.User.objects.all()
    else:
        user_list = modify.models.User.objects.filter(username=username)
    return render(request, "userinfo.html", {"user_list": user_list})##将数据导入html模板中，进行数据渲染。
def add(request):
    if request.method == 'GET':
        return render(request,'add.html')
    else:
        username = request.POST.get('username','')
        password = request.POST.get('password','')
        password = take_md5(password)
        phone = request.POST.get('phone','')
        email = request.POST.get('email','')
        type = request.POST.get('type','')
        modify.models.User.objects.create(username=username, password=password, phone=phone, email=email, type=type)
        userLogin.models.User.objects.create(username=username, password=password, phone=phone, email=email, type=type)
        return redirect("/userinfo")



def delete(request):
    name = request.GET.get('username') #根据用户名删除用户
    modify.models.User.objects.filter(username=name).delete()
    userLogin.models.User.objects.filter(username=name).delete()
    return redirect("/userinfo") #删除之后回到/userinfo/界面

def edit(request):
    username = request.GET.get('username')
    user_data = User.objects.get(username=username)
    if request.method == 'GET':
        return render(request, "edit.html", {"user_data":user_data})

    password = request.POST.get('password')
    password = take_md5(password)
    phone = request.POST.get('phone')
    email = request.POST.get('email')
    type = request.POST.get('type')
    modify.models.User.objects.filter(username=username).update(password=password, phone=phone, email=email, type=type)
    userLogin.models.User.objects.filter(username=username).update(password=password, phone=phone, email=email, type=type)
    return redirect("/userinfo")

def index(request):
    return render(request, "index.html")

