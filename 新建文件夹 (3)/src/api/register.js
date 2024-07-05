import request from '@/utils/request'
// 假设您已经安装并导入了一个HTTP请求库，比如axios
import axios from 'axios';

// 定义一个函数来发送请求
export function registerUser(data) {
  // 构建请求URL
  const url = '/api/register'; // 替换为实际的API端点

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
