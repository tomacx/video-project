from django.shortcuts import render
from userLogin.models import User


# Create your views here.
#呃呃 反正我两个功能搞好了 再看前端连接
def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username)
        corr_user = User.objects.filter(username=username).first()
        if username == corr_user.username and password == corr_user.password:
            #return render(request, './course/work_choose.html', {'USER_ID': sno}) 返回成功登录界面
            return render(request, 'success.html')
    return render(request, 'login.html')


def register(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    u = User.objects.create(username=username, password=password)
    print(username, password)
    u.save()
