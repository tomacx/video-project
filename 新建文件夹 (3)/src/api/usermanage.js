import request from '@/utils/request'

export function login(data) {
  return request({
    url: '/vue-admin-template/user/login',
    method: 'post',
    data
  })
}

export default  {
  getuserList(searchModel){
    return request({
      url:'/vue-admin-template/user/list',
      method:'get',
      params:{
        pageNo:searchModel.pageNo,
        pagesize:searchModel.pagesize,
        username:searchModel.username,
        phone:searchModel.phone

      }
    });
  },
  addUser(user){
    return request({
      url:'/user',
      method:'post',
      data:user
    });
  },
}


