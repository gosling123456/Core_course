#include <cstdio>
#include <malloc.h>
#include <iostream>
#include <windows.h>
using namespace std;
typedef struct {
    int * pBase;//描述的数组第一个元素地址
    int len;//描述数组长度
    int cnt;//当前数组有效长度
} Arr;
void init_arr(Arr * pArr,int length);//初始化顺序表
bool insert_arr(Arr * pArr,int pos, int val);//像顺序表插入元素
bool append_arr(Arr * pArr,int val);//像顺序表追加元素
bool delete_arr(Arr * pArr,int pos, int * val);//删除顺序表元素
bool Is_empty(Arr * pArr);//判空
bool is_full(Arr * pArr);//判满
void sort_arr(Arr * pArr);//排序元素
void show_arr(Arr * pArr);//显示元素
bool inversion_arr(Arr * pArr);//反转
bool free_arr(Arr * pArr);//释放
int main(){
    Arr arr;
    int a ,choose;
    int val;
    cout<<"请输入预计元素个数：";
    cin>>a;
    init_arr(&arr, a);//初始化
    while (true){
        cout<<"\t\t1.退出\n\t\t2.插入\n\t\t3.追加\n\t\t4.删除\n\t\t5.排序\n\t\t6.显示\n\t\t7.反转\n\t\t8.释放"<<endl;
        cout<< "请选择你项进行的操作:";
        cin>>choose;
        switch (choose) {
            case 1: {
                exit(-1);
            }
            case 2:{
                int data,p;
                cout<<"请输入你想插入的位置：";
                cin>>p;
                cout<<"请输入你要插入的数据：";
                cin>>data;
                insert_arr(&arr,p, data);
                break;
            }
            case 3:{
                int data;
                cout<<"请输入你要追加的数据：";
                cin>>data;
                append_arr(&arr, data);
                break;
            }
            case 4:{
                int data,p;
                cout<<"请输入你想删除的位置：";
                cin>>p;
                delete_arr(&arr,p, &val);
                cout<<"删除的数据为："<<val<<endl;
                break;
            }
            case 5: {
                sort_arr(&arr);
                break;
            }
            case 6: {
                show_arr(&arr);
                break;
            }
            case 7: {
                inversion_arr(&arr);
                break;
            }
            case 8:{
                free_arr(&arr);
                break;}
            default:
                cout<<"输入不合法！"<<endl;
        }
    }

}
void init_arr(Arr * pArr,int length)//初始化
{
    //像系统申请(sizeof(int)*length)字节的空间
    pArr->pBase = (int *) malloc(sizeof(int)*length);
    if(NULL == pArr->pBase){
        printf("动态内存分配失败！\n");
        Sleep(1000);
    }
    else{
        pArr->len =length;
        pArr->cnt = 0;
        printf("动态内存分配成功！\n");
        Sleep(1000);
    }

}
bool append_arr(Arr * pArr,int value){
    if(is_full(pArr)){
        cout<<"顺序表已满，不能追加！"<<endl;
        Sleep(1000);
        return false;
    }
    pArr->pBase[pArr->cnt] = value;
    (pArr->cnt)++;
    printf("添加成功！\n");
    Sleep(1000);
    return true;
}
bool Is_empty(Arr * pArr){
    if(pArr->cnt==0) return true;//空
    else return false;
}
bool is_full(Arr * pArr){
    if(pArr->cnt==pArr->len) return true;
    else return false;
}
void  show_arr(Arr * pArr){
    if(Is_empty(pArr)){
        printf("空表!\n");
        Sleep(1000);
        exit(-1);
    }else{
        for(int i=0;i<pArr->cnt;i++){
            printf("%d\n",pArr->pBase[i]);
        }
        Sleep(1000);}
}
bool insert_arr(Arr * pArr,int pos,int val){//pos位置插，pos表示下标
    if(is_full(pArr)){
        printf("表已满，不能插入!\n");
        Sleep(1000);
        return false;
    }
    else if(pos<0||pos>pArr->cnt){
        printf("插入非法!\n");
        Sleep(1000);
        return false;
    }
    for(int i=pArr->cnt;i>=pos;i--){
        pArr->pBase[i+1] = pArr->pBase[i];
    }
    pArr->pBase[pos] = val;
    pArr->cnt++;
    cout<<"插入成功！"<<endl;
    Sleep(1000);
    return true;
}
bool delete_arr(Arr * pArr,int pos, int * val){
    if(Is_empty(pArr)){
        printf("空表不能删除!\n");
        Sleep(1000);
        exit(-1);
    }else if(pos<0||pos>pArr->cnt-1){
        printf("删除非法!\n");
        Sleep(1000);
        return false;
    }
    *val = pArr->pBase[pos];//val表示被删元素地址
    for(int i=pos+1;i<=pArr->cnt-1;i++){
        pArr->pBase[i-1] = pArr->pBase[i];
    }pArr->cnt--;
    return true;
}
void sort_arr(Arr * pArr){
    for(int i=0;i<pArr->cnt-1;i++) {
        for(int j=0;j<pArr->cnt-1-i;j++){
            if(pArr->pBase[j]>pArr->pBase[j+1]){
                int t=pArr->pBase[j];
                pArr->pBase[j] = pArr->pBase[j+1];
                pArr->pBase[j+1] = t;
            }
        }
    }
    cout<<"排序成功！"<<endl;
    Sleep(1000);
}
bool inversion_arr(Arr * pArr){
    int i=0;
    int j=pArr->cnt-1;
    while(i<j){
        int t = pArr->pBase[i];
        pArr->pBase[i]=pArr->pBase[j];
        pArr->pBase[j] = t;
        i++;j--;
    }
    cout<<"反转成功！"<<endl;
    Sleep(1000);
    return true;
}
bool free_arr(Arr * pArr){
    free(pArr->pBase);
    pArr->pBase=NULL;
    pArr->len=0;
    pArr->cnt=0;
    cout<<"释放成功！"<<endl;
    Sleep(1000);
    return true;
}