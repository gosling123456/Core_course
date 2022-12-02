# #!/usr/bin/env python
# #-*- coding:utf-8 -*-
#
# import os, sys
# from tkinter import *
# from tkinter.font import Font
# from tkinter.ttk import *
# #Usage:showinfo/warning/error,askquestion/okcancel/yesno/retrycancel
# from tkinter.messagebox import *
# #Usage:f=tkFileDialog.askopenfilename(initialdir='E:/Python')
# #import tkinter.filedialog as tkFileDialog
# #import tkinter.simpledialog as tkSimpleDialog  #askstring()
#
# class Application_ui(Frame):
#     #这个类仅实现界面生成功能，具体事件处理代码在子类Application中。
#     def __init__(self, master=None):
#         Frame.__init__(self, master)
#         self.master.title('Form1')
#         self.master.geometry('1100x482')
#         self.createWidgets()
#
#     def createWidgets(self):
#         self.top = self.winfo_toplevel()
#
#         self.style = Style()
#
#         self.MainMenu = Menu(self.top, tearoff=0)
#
#         self.j = Menu(self.MainMenu, tearoff=0)
#         self.j.add_command(label='最新', command=self.jx_Cmd)
#         self.j.add_command(label='最热', command=self.hot_Cmd)
#         self.MainMenu.add_cascade(menu=self.j, label='精选')
#
#         self.zt = Menu(self.MainMenu, tearoff=0)
#         self.zt.add_command(label='经典', command=self.jd_Cmd)
#         self.zt.add_command(label='轻音乐', command=self.qinyy_Cmd)
#         self.zt.add_command(label='怀旧', command=self.huijiu_Cmd)
#         self.zt.add_command(label='抖音', command=self.dy_Cmd)
#         self.zt.add_command(label='古风', command=self.gufen_Cmd)
#         self.zt.add_command(label='网络', command=self.wangluo_Cmd)
#         self.zt.add_command(label='游戏', command=self.youxi_Cmd)
#         self.zt.add_command(label='影视', command=self.yinshi_Cmd)
#         self.zt.add_command(label='佛乐', command=self.fuyv_Cmd)
#         self.zt.add_command(label='网红', command=self.wanghong_Cmd)
#         self.zt.add_command(label='纯音乐', command=self.congyy_Cmd)
#         self.zt.add_command(label='ACG', command=self.ACG_Cmd)
#         self.zt.add_command(label='3D', command=self.D3_Cmd)
#         self.zt.add_command(label='KTV', command=self.ktv_Cmd)
#         self.zt.add_command(label='器乐', command=self.qiyue_Cmd)
#         self.zt.add_command(label='翻唱', command=self.fancang_Cmd)
#         self.zt.add_command(label='喊麦', command=self.hanmai_Cmd)
#         self.zt.add_command(label='店铺专用', command=self.dianpu_Cmd)
#         self.MainMenu.add_cascade(menu=self.zt, label='主题')
#
#         self.x = Menu(self.MainMenu, tearoff=0)
#         self.x.add_command(label='伤感', command=self.shanggan_Cmd)
#         self.x.add_command(label='放松', command=self.fangsong_Cmd)
#         self.x.add_command(label='励志', command=self.lizhi_Cmd)
#         self.x.add_command(label='开心', command=self.kaixing_Cmd)
#         self.x.add_command(label='甜蜜', command=self.tianmi_Cmd)
#         self.x.add_command(label='兴奋', command=self.xingfen_Cmd)
#         self.x.add_command(label='安静', command=self.anjing_Cmd)
#         self.x.add_command(label='治愈', command=self.zhiyu_Cmd)
#         self.x.add_command(label='寂寞', command=self.jimo_Cmd)
#         self.x.add_command(label='思念', command=self.sinian_Cmd)
#         self.MainMenu.add_cascade(menu=self.x, label='心情')
#
#         self.cj = Menu(self.MainMenu, tearoff=0)
#         self.cj.add_command(label='开车', command=self.kaiche_Cmd)
#         self.cj.add_command(label='运动', command=self.yundong_Cmd)
#         self.cj.add_command(label='睡前', command=self.shuiqian_Cmd)
#         self.cj.add_command(label='跳舞', command=self.tiaowu_Cmd)
#         self.cj.add_command(label='学习', command=self.xuexi_Cmd)
#         self.cj.add_command(label='清晨', command=self.qingchen_Cmd)
#         self.cj.add_command(label='夜店', command=self.yandian_Cmd)
#         self.cj.add_command(label='校园', command=self.xiaoyuan_Cmd)
#         self.cj.add_command(label='咖啡店', command=self.kafeidian_Cmd)
#         self.cj.add_command(label='旅行', command=self.lvxing_Cmd)
#         self.cj.add_command(label='工作', command=self.gongzuo_Cmd)
#         self.cj.add_command(label='广场舞', command=self.guangchuangwu_Cmd)
#         self.MainMenu.add_cascade(menu=self.cj, label='场景')
#
#         self.nd = Menu(self.MainMenu, tearoff=0)
#         self.nd.add_command(label='70后', command=self.h70_Cmd)
#         self.nd.add_command(label='80垢', command=self.h80_Cmd)
#         self.nd.add_command(label='90后', command=self.h90_Cmd)
#         self.nd.add_command(label='00后', command=self.h00_Cmd)
#         self.nd.add_command(label='10后', command=self.h10_Cmd)
#         self.MainMenu.add_cascade(menu=self.nd, label='年代')
#
#         self.qf = Menu(self.MainMenu, tearoff=0)
#         self.qf.add_command(label='流行', command=self.liuxin_Cmd)
#         self.qf.add_command(label='电子', command=self.dianzi_Cmd)
#         self.qf.add_command(label='摇滚', command=self.yaoguong_Cmd)
#         self.qf.add_command(label='民歌', command=self.minge_Cmd)
#         self.qf.add_command(label='民谣', command=self.migyao_Cmd)
#         self.qf.add_command(label='古典', command=self.gudian_Cmd)
#         self.qf.add_command(label='嘻哈', command=self.xiha_Cmd)
#         self.qf.add_command(label='乡村', command=self.xiangcun_Cmd)
#         self.qf.add_command(label='爵士', command=self.jueshi_Cmd)
#         self.qf.add_command(label='R＆B', command=self.rb_Cmd)
#         self.MainMenu.add_cascade(menu=self.qf, label='曲风')
#
#         self.yy = Menu(self.MainMenu, tearoff=0)
#         self.yy.add_command(label='华语', command=self.huanyu_Cmd)
#         self.yy.add_command(label='欧美', command=self.oumei_Cmd)
#         self.yy.add_command(label='韩语', command=self.hanyv_Cmd)
#         self.yy.add_command(label='粤语', command=self.yueyu_Cmd)
#         self.yy.add_command(label='日语', command=self.riyv_Cmd)
#         self.yy.add_command(label='小语种', command=self.xiaoyuzhong_Cmd)
#         self.MainMenu.add_cascade(menu=self.yy, label='语言')
#
#         self.aboutme = Menu(self.MainMenu, tearoff=0)
#         self.aboutme.add_command(label='使用说明', command=self.abouthelp_Cmd)
#         self.aboutme.add_command(label='关于', command=self.aboutver_Cmd)
#         self.MainMenu.add_cascade(menu=self.aboutme, label='帮助')
#         self.top['menu'] = self.MainMenu
#
#
# class Application(Application_ui):
#     #这个类实现具体的事件处理回调函数。界面生成代码在Application_ui中。
#     def __init__(self, master=None):
#         Application_ui.__init__(self, master)
#
#     def jx_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def hot_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def jd_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def qinyy_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def huijiu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def dy_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def gufen_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def wangluo_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def youxi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def yinshi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def fuyv_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def wanghong_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def congyy_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def ACG_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def D3_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def ktv_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def qiyue_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def fancang_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def hanmai_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def dianpu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def shanggan_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def fangsong_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def lizhi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def kaixing_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def tianmi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def xingfen_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def anjing_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def zhiyu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def jimo_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def sinian_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def kaiche_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def yundong_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def shuiqian_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def tiaowu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def xuexi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def qingchen_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def yandian_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def xiaoyuan_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def kafeidian_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def lvxing_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def gongzuo_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def guangchuangwu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def h70_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def h80_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def h90_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def h00_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def h10_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def liuxin_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def dianzi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def yaoguong_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def minge_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def migyao_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def gudian_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def xiha_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def xiangcun_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def jueshi_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def rb_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def huanyu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def oumei_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def hanyv_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def yueyu_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def riyv_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def xiaoyuzhong_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def abouthelp_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
#     def aboutver_Cmd(self, event=None):
#         #TODO, Please finish the function here!
#         pass
#
# if __name__ == "__main__":
#     top = Tk()
#     Application(top).mainloop()
#     try: top.destroy()
#     except: pass
#
# from tkinter import *
# root = Tk()
# # 设置窗口的背景颜色以区别画布
# root.config(bg='#87CEEB')
# root.title("202018526高树林的凸包可视化")
# cv=Canvas(root,bg="white",width =800, height = 800)
# cv.pack()
# Button(root,text='高树林，',command=root.quit).pack()
#
# #坐标轴线
# cv.create_line(100,500,700,500,width=2)
# cv.create_line(100,500,100,50,width=2)
#
# #x轴
# for i in range(11):
#     x = 100 + (i*60)
#     cv.create_line(x,500,x,495,width=2)
#     cv.create_text(x, 499, text='%d'%(10+i),anchor=N)
#
# #y轴
# for i in range(6):
#     y = 500 - (i*80)
#     cv.create_line(100,y,105,y,width=2)
#     cv.create_text(96,y, text='%5.f'%(50*i),anchor=E)
# scaled = []
# for x,y in [(1,1),(2,2)]:
#     scaled.append((100+6*x,500-8*y/5))
# cv.create_line(scaled,fill='green')
# for x,y in scaled:
#     cv.create_oval(x-6,y-6,x+6,y+6,width=1,outline='black',fill='blue')
# cv.create_text(350,560,text='使用',fill='red',font='SimHei 20 bold')
# root.mainloop()
import tkinter as tk
from tkinter.simpledialog import askstring, askinteger, askfloat
# 接收一个整数
def print_integer():
  res = askinteger("Spam", "Egg count", initialvalue=12*12)
  print(res)
root = tk.Tk()
tk.Button(root, text='取一个整数', command=print_integer).pack()

root.mainloop()
