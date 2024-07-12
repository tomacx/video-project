import pickle
import time
from urllib import request
import json

from captcha.helpers import captcha_image_url
from captcha.models import CaptchaStore
from django.http import HttpResponse
from django.shortcuts import render, redirect

import modify
from userLogin.models import User
import hashlib
from django.template import RequestContext
from userLogin.form import UserLogin,UserRegister
from modify import models

from django.core import signing
import hashlib
from django.core.cache import cache

HEADER = {'typ': 'JWP', 'alg': 'default'}
KEY = 'CHEN_FENG_YAO'
SALT = 'www.lanou3g.com'
TIME_OUT = 30 * 60  # 30min
USERNAME = ''
# Create your views here.
def take_md5(content):
    content_bytes = content.encode('utf-8') #将字符串转为字节对象
    hash = hashlib.md5()    #创建hash加密实例
    hash.update(content_bytes)    #hash加密
    result = hash.hexdigest()  #得到加密结果
    return result[:16]

def captcha():
    hashkey = CaptchaStore.generate_key()   #验证码答案
    image_url = captcha_image_url(hashkey)  #验证码地址
    captcha = {'hashkey': hashkey, 'image_url': image_url}
    return captcha
#刷新验证码
def refresh_captcha(request):
    return HttpResponse(json.dumps(captcha()), content_type='application/json')
# 验证验证码
def jarge_captcha(captchaStr, captchaHashkey):
    if captchaStr and captchaHashkey:
        try:
            # 获取根据hashkey获取数据库中的response值
            get_captcha = CaptchaStore.objects.get(hashkey=captchaHashkey)
            if get_captcha.response == captchaStr.lower():     # 如果验证码匹配
                return True
        except:
            return False
    else:
        return False

def encrypt(obj):#加密
    value = signing.dumps(obj, key=KEY, salt=SALT)
    value = signing.b64_encode(value.encode()).decode()
    return value

def decrypt(src):#解密
    src = signing.b64_decode(src.encode()).decode()
    raw = signing.loads(src, key=KEY, salt=SALT)
    print(type(raw))
    return raw

def create_token(username):
    header = encrypt(HEADER)
    payload = {"username": username, 'iat': time.time()}
    payload = encrypt(payload)
    md5 = hashlib.md5()
    md5.update(("%s.%s"%(header,payload)).encode())
    signature = md5.hexdigest()
    token = "%s.%s.%s" % (header,payload,signature)
    cache.set(username, token, TIME_OUT)
    return token

def get_payload(token):
    payload = str(token).split('.')[1]
    payload = decrypt(payload)
    return payload

def get_username(token):
    payload = get_payload(token)
    return payload['username']
    pass

def check_token(token):
    username = get_username(token)
    last_token = cache.get(username)
    if last_token:
        return last_token == token
    return False

def login(request):
    # request.session['TOKEN'] = ".."
    if request.method == 'POST':
        form = UserLogin(request.POST)
        capt = request.POST.get("captcha", None)  # 用户提交的验证码
        key = request.POST.get("hashkey", None)  # 验证码答案
        if jarge_captcha(capt, key):
            if form.is_valid():
                username = request.POST.get('username')
                password = request.POST.get('password')
                password = take_md5(password)
                namefilter = User.objects.filter(username=username,password=password)
                print(username)
                if len(namefilter) > 0 and namefilter[0].type == "管理员":
                    request.session['USERNAME'] = username  # 保存数据到会话
                    token = create_token(username)
                    request.session['TOKEN'] = token
                    print(token)



                    return redirect("/index")
                elif len(namefilter) > 0 and namefilter[0].type == "场务人员":
                    request.session['USERNAME'] = username  # 保存数据到会话
                    token = create_token(username)
                    request.session['TOKEN'] = token

                    return redirect("/index_worker")
                else:
                    form = UserLogin()
                    hashkey = CaptchaStore.generate_key()  # 验证码答案
                    image_url = captcha_image_url(hashkey)  # 验证码地址
                    captcha = {'hashkey': hashkey, 'image_url': image_url}

                    return render(request, 'login.html', locals())
            else:
                form = UserLogin()
                hashkey = CaptchaStore.generate_key()  # 验证码答案
                image_url = captcha_image_url(hashkey)  # 验证码地址
                captcha = {'hashkey': hashkey, 'image_url': image_url}

                return render(request,'login.html', locals())
        else:
            hashkey = CaptchaStore.generate_key()  # 验证码答案
            image_url = captcha_image_url(hashkey)  # 验证码地址
            captcha = {'hashkey': hashkey, 'image_url': image_url}
            return render(request, 'login.html', locals())

    else:
        hashkey = CaptchaStore.generate_key()  # 验证码答案
        image_url = captcha_image_url(hashkey)  # 验证码地址
        captcha = {'hashkey': hashkey, 'image_url': image_url}
        return render(request,'login.html', locals())

def register(request):
    if request.method == 'POST':
        form = UserRegister(request.POST)
        capt = request.POST.get("captcha", None)  # 用户提交的验证码
        key = request.POST.get("hashkey", None)  # 验证码答案
        if jarge_captcha(capt, key):

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

                        hashkey = CaptchaStore.generate_key()  # 验证码答案
                        image_url = captcha_image_url(hashkey)  # 验证码地址
                        captcha = {'hashkey': hashkey, 'image_url': image_url}
                        return render(request, 'login.html', locals())
            form = UserRegister()
            hashkey = CaptchaStore.generate_key()  # 验证码答案
            image_url = captcha_image_url(hashkey)  # 验证码地址
            captcha = {'hashkey': hashkey, 'image_url': image_url}

            return render(request,'register.html', locals())
        else:
            hashkey = CaptchaStore.generate_key()  # 验证码答案
            image_url = captcha_image_url(hashkey)  # 验证码地址
            captcha = {'hashkey': hashkey, 'image_url': image_url}

            return render(request, 'register.html', locals())

    else:
        hashkey = CaptchaStore.generate_key()  # 验证码答案
        image_url = captcha_image_url(hashkey)  # 验证码地址
        captcha = {'hashkey': hashkey, 'image_url': image_url}

        return render(request,'register.html', locals())

def index(request):
    return render(request,'index.html')