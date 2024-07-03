<template>
    <div>
      <!--搜索栏-->
      <el-card id="search">
        <el-input v-model="searchModel.username" placeholder="姓名"></el-input>
        <el-input v-model="searchModel.phone" placeholder="电话"></el-input>
        <el-button type="primary" round>查询</el-button>
        <el-button @click="openEditUI" type="primary" icon="el-icon-plus" circle></el-button>
      </el-card>
      <!--结果栏-->
      <el-card>
        <el-table
    :data="userList"
    stripe
    style="width: 100%">
    <el-table-column
      prop="index"
      label="#"
      width="80">
    </el-table-column>
    <el-table-column
      prop="username"
      label="姓名"
      width="180">
    </el-table-column>
    <el-table-column
      prop="phone"
      label="电话"
      width="180">
    </el-table-column>
    <el-table-column
      prop="mail"
      label="电子邮件">
    </el-table-column>
    <el-table-column
      
      label="操作"
      width="180">
    </el-table-column>
  </el-table>
      </el-card>
      <!--分页组件-->
      <el-pagination
      @size-change="handleSizeChange"
      @current-change="handleCurrentChange"
      :current-page="searchModel.pageNo"
      :page-sizes="[5, 10, 15, 50]"
      :page-size="searchModel.pageSize"
      layout="total, sizes, prev, pager, next, jumper"
      :total="total">
    </el-pagination>
    <!--用户信息编辑对话框-->
    <el-dialog :title="title" :visible.sync="dialogFormVisible">
  <el-form :model="userForm">
    <el-form-item label="姓名" :label-width="formLabelWidth">
      <el-input v-model="userForm.name" autocomplete="off"></el-input>
    </el-form-item>
    
  </el-form>
  <div slot="footer" class="dialog-footer">
    <el-button @click="dialogFormVisible = false">取 消</el-button>
    <el-button type="primary" @click="dialogFormVisible = false">确 定</el-button>
  </div>
</el-dialog>
    </div>
  </template>
   <!--定义的查询对象-->
  <script>
  export default {
    data(){
     return {
      formLabelWidth:'130px',
      userForm:{},
      dialogFormVisible:false,
      title:"",
      total:0,
      searchModel:{
        pageNo:1,
        pageSize:10
      },
      userList:[]
     }
      
    },
    methods:{
      openEditUI:{
        //  this.title='新增人员';
        //  this.dialogFormVisible=true;
      
      },
      handleSizeChange:{

      },
      handleCurrentChange:{
        
      }
    }
  }
  </script>
  
  <style>
    #search .el-input{
      width: 200px;
    }
  </style>
  