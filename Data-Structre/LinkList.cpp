//排序有问题"
#include <stdio.h>
#include <iostream>
#include <malloc.h>
#include <windows.h>
using namespace std;

typedef struct Node{
    int score;//数据域
    struct Node * pNext;//指针域
}NODE,*PNODE;//NODE等价于struct Node, PNODE等价于struct Node *
PNODE create_list_tail();//创建单链表并返回头指针
PNODE create_list_head();
void travels(PNODE pHead);
bool Is_empty(PNODE pHead);
bool insert(PNODE pHead);//pos表示插入的位置，val表示插入的值
bool delete_list(PNODE pHead);//删除元素
int length(PNODE pHead);//求表长
void upsort(PNODE pHead);//升序
void downsort(PNODE pHead);//降序
bool freeList(PNODE pHead);//清空
void menu(PNODE pHead);//菜单
int main(){
    PNODE pHead=nullptr;
    int choose;
    cout<<"0.头插法\n"<<"1.尾插法\n"<<"选择建表方式：";
    cin>>choose;
    switch ((choose)) {
        case 0:{pHead = create_list_head();break;}
        case 1:{pHead = create_list_tail();break;}
    }
    while(true)
    menu(pHead);
}
PNODE create_list_tail(){//尾插法
    int len,val;//len为有效节点个数
    PNODE pHead=(PNODE)malloc(sizeof(NODE));
    PNODE pTail=pHead;//尾指针，用于插入，当前指向的是头结点
    if(pHead==nullptr){
        cout<<"空间分配失败！"<<endl;
        exit(-1);
    }
    cout<<"请输入要申请的链表结点个数：";
    cin>>len;
    for(int i=0;i<len;i++){
        cout<<"请输入第"<<i+1<<"个结点的值：";
        cin>>val;
        PNODE pNew = (PNODE)malloc((sizeof(NODE)));//申请给新节点分配空间
        if(pNew==nullptr){
            cout<<"空间分配失败，程序终止！"<<endl;
            exit(-1);
        }
        pNew->score = val;//给数据域赋值
        pTail->pNext = pNew;//将新的结点连接在pTail后面
        pNew->pNext = nullptr;//相当与链表就这么长，pNew的指针域没有内容指向
        pTail = pNew;//相当与pTail向下移动，始终指向尾节点
    }

    return pHead;
}
PNODE create_list_head(){//头插法
    int len,val;//len为有效结点个数
    PNODE p=(PNODE)malloc(sizeof(NODE));
    p->pNext=nullptr;//一定要将首元节点的指针域置为空，否则默认不为空输出会出错
    if(p==NULL){
        cout<<"空间分配失败！"<<endl;
        exit(-1);
    }
    cout<<"请输入要申请的链表结点个数：";
    cin>>len;
    for(int i=len-1;i>=0;i--){
        cout<<"请输入结点的值：";
        cin>>val;
        PNODE pNew = (PNODE)malloc((sizeof(NODE)));//申请给新节点分配空间
        if(pNew==nullptr){
            cout<<"空间分配失败，程序终止！"<<endl;
            exit(-1);
        }
        pNew->score = val;//给数据域赋值
        pNew->pNext = p->pNext;
        p->pNext = pNew;//相当pHead向上移动，始终指向尾节点
    }
    return p;
}
void travels(PNODE pHead){
    if(Is_empty(pHead)){
        cout<<"空表！"<<endl;
        return;
    }
    PNODE p= pHead->pNext;//p指向首元节点
    cout<<"链表元素为："<<endl;
    while(p!=nullptr){
        cout<<p->score<<endl;
        p=p->pNext;
    }
}
bool Is_empty(PNODE pHead){
    if(pHead->pNext==nullptr){
        return true;
    }return false;
}
bool insert(PNODE pHead){//尾插法
    int pos,val;
    cout<<"请输入你要插入下标的位置：";
    cin>>pos;
    cout<<"请输入要插入的值：";
    cin>>val;
    PNODE s =pHead;
    int i=0;
    if(pos<0||pos> length(pHead)||s==nullptr){
        cout<<"插入失败！"<<endl;
        return false;
    }
    while(s!=nullptr&&i<pos){//while停止循环时，s指针停靠在要插入结点的上一个结点
        s=s->pNext;
        i++;
    }
    PNODE  pNew=(PNODE)malloc(sizeof(NODE));
    pNew->score = val;
    pNew->pNext = s->pNext;
    s->pNext = pNew;
    cout<<"插入成功！"<<endl;
    return true;
}
bool delete_list(PNODE pHead){
    //删除第pos位置的元素
    int pos;
    int * pval;
    cout<<"请输入删除元素的下标位置：";
    cin>>pos;
    PNODE p=pHead,q;
    if(Is_empty(p)){
        cout<<"空链表不能删除元素！"<<endl;
        return false;
    }
    if(pos<0||p->pNext==nullptr||pos> length(pHead)){
        cout<<"链表删除失败！"<<endl;
        return false;
    }
    for(int i=0;i<pos;i++){
        p=p->pNext;
    }
    q = p->pNext;
    p->pNext = p->pNext->pNext;
    cout<<"删除的元素为"<<q->score<<endl;
    free(q);
    return true;
}
int length(PNODE pHead){
    PNODE  p=pHead->pNext;
    int len = 0;
    if(Is_empty(pHead)) return 0;
    while(p!=NULL){
        len++;
        p = p->pNext;
    }return len;
}
void downsort(PNODE pHead){
    PNODE p = pHead->pNext,first=p,second=first,end=NULL;
    while(first!=end){
        while(first->pNext!=end){
            if(first->score < first->pNext->score){
                int temper = first->score;
                first->score = first->pNext->score;
                first->pNext->score = temper;
            }
            first = first->pNext;
        }
        end = first;
        first = p;
    }
    cout<<"排序成功！"<<endl;
}
void upsort (PNODE pHead)
{
    PNODE p = pHead->pNext,first=p,second=first,end=NULL;
    while(first!=end){
        while(first->pNext!=end){
            if(first->score > first->pNext->score){
                int temper = first->score;
                first->score = first->pNext->score;
                first->pNext->score = temper;
            }
            first = first->pNext;
        }
        end = first;
        first = p;
    }
cout<<"排序成功！"<<endl;
}
bool freeList(PNODE pHead){
    pHead->pNext = NULL;
    cout<<"链表清空成功！"<<endl;
    Sleep(500);
    return true;
}
//bool destory(PNODE pHead){
//    free(pHead);
//}将空间释放给内存，无实际意义。
bool inversion(PNODE pHead){//inversion:反转
    if(pHead==nullptr||pHead->pNext==nullptr){
        cout<<"空表不能逆转！"<<endl;
        return false;
    }
    int val;
    PNODE p=pHead->pNext;
    PNODE s=(PNODE)malloc(sizeof(NODE));
    s->pNext=nullptr;//一定要将首元节点的指针域置为空，否则默认不为空输出会出错
    if(p==nullptr){
        cout<<"空间分配失败！"<<endl;
        exit(-1);
    }
    while(p){
        val = p->score;
        PNODE pNew = (PNODE)malloc((sizeof(NODE)));//申请给新节点分配空间
        if(pNew==nullptr){
            cout<<"空间分配失败，程序终止！"<<endl;
            exit(-1);
        }
        pNew->score = val;//给数据域赋值
        pNew->pNext = s->pNext;
        s->pNext = pNew;//相当pHead向上移动，始终指向尾节点
        p = p->pNext;
    }
    pHead->pNext=s->pNext;
    cout<<"反转成功！"<<endl;
    return true;
}
void menu(PNODE pHead){
    system("cls");
    int choose;
    cout<<"\t\t1.退出\n\t\t2.插入\n\t\t3.删除\n\t\t4.排序\n\t\t5.显示\n\t\t6.反转\n\t\t7.清空"<<endl;
    cout<<"请选择操作方式：";
    cin>>choose;
    switch (choose) {
        case 1:{cout<<"谢谢使用!"<<endl;exit(-1);}
        case 2:{insert(pHead);break;}
        case 3:{
            delete_list(pHead);
            break;}
        case 4:{
            cout<<"\t\t0.升序\n\t\t1.降序"<<endl<<"请选择排序方式：";
            cin>>choose;
            if(choose==0) upsort(pHead);
            if(choose==1) downsort(pHead);
            break;
            }
        case 5:{
            travels(pHead);break;}
        case 6:{
            inversion(pHead);
            break;}
        case 7:{
            freeList(pHead);
            break;}
//        case 8:{
//                destory(pHead);
//                break;
//            }
        default:{cout<<"键入非法！"<<endl;break;}

    }
}
