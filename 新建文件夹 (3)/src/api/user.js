import request from '@/utils/request'

export function login(data) {
  return request({
    url: '/vue-admin-template/user/login',
    method: 'post',
    data
  })
}

export function getInfo(token) {
  return request({
    url: '/vue-admin-template/user/info',
    method: 'get',
    params: { token }
  })
}

export function logout() {
  return request({
    url: '/vue-admin-template/user/logout',
    method: 'post'
  })
}



// 定义一个函数来发送请求
export function registerUser(data) {
  // 构建请求URL
  const url = '/vue-admin-template/user/register'; // 替换为实际的API端点

  // 发送POST请求
  return axios.post(url, null, {
    params: {
      username: data.username,
      password1: data.password1,
      password2: data.password2,
      email: data.email,
      phone: data.phone,
      type: data.type
    }
  });
}