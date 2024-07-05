<template>
    <div>
      <!-- 用户注册表单 -->
      <form @submit.prevent="handleRegister">
        <!-- 表单输入 -->
        <input v-model="registrationData.username" placeholder="Username" />
        <input v-model="registrationData.password1" type="password" placeholder="Password" />
        <input v-model="registrationData.password2" type="password" placeholder="Confirm Password" />
        <input v-model="registrationData.email" placeholder="Email" />
        <input v-model="registrationData.phone" placeholder="Phone" />
        <input v-model="registrationData.type" placeholder="Type" />
        <!-- 注册按钮 -->
        <button type="submit">Register</button>
      </form>
    </div>
  </template>
  
  <script>
  import { registerUser } from '@/path/to/your/request/util';
  
  export default {
    data() {
      return {
        registrationData: {
          username: '',
          password1: '',
          password2: '',
          email: '',
          phone: '',
          type: ''
        }
      };
    },
    methods: {
      async handleRegister() {
        try {
          // 调用registerUser函数并传递表单数据
          const response = await registerUser(this.registrationData);
          // 处理响应
          console.log(response.data);
          // 可能还需要其他逻辑，比如重定向或显示成功消息
        } catch (error) {
          // 处理错误
          console.error(error);
          // 可能还需要其他逻辑，比如显示错误消息
        }
      }
    }
  };
  </script>
  