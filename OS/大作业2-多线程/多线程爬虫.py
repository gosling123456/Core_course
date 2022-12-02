import requests
from lxml import etree
import threading, psutil, os
import time
from concurrent.futures import ThreadPoolExecutor

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3863.400 QQBrowser/10.8.4334.400',
}


# 定义线程函数
def call_back(res):
    # 完成线程的操作，进行回调的函数
    res = res.result()  # 获取结果
    # return res


def downmp4(video_url, video_name):
    path = '包图网大国工匠视频多线程下载'
    # 对当前目录下面的文件夹进行判断，如果没有自动创建一个文件夹拉存储
    if not os.path.exists(path):
        os.mkdir(path)
    # 得到进程线程相关的信息
    thread = threading.current_thread()  # 得到当前的线程对象
    process = psutil.Process(os.getpid())  # 得到当前的进程对象
    print(thread.ident, thread.name, process.pid, process.name())
    start_time = time.time()
    video_url = "https:" + video_url
    video_name = video_name.strip().replace("<strong>", "").replace("</strong>", "")
    video_content = requests.get(url=video_url, headers=headers).content
    with open(path + './%s.mp4' % video_name, "wb") as f:
        f.write(video_content)
    finish_time = time.time() - start_time
    return "每个视频下载的时间" + str(finish_time)


def parsePage():
    url = "https://ibaotu.com/tupian/gongjiangjingshen/7-0-0-0-0-0-0.html?format_type=0"  # 第一步，确定爬虫地址
    response = requests.get(url=url, headers=headers)  # 第二步：发送请求
    html_content = response.text  # 第三步：获取数据
    tree = etree.HTML(html_content)
    video_page_url = tree.xpath("//ul/li/div/div/a/div[1]/video/@src")
    print('爬虫开始，一共要爬取%d个视频.......' % len(video_page_url))
    video_name = tree.xpath("//ul/li/@pr-data-title")
    start_time = time.time()
    executor = ThreadPoolExecutor(2)  # 1.创建线程池，初始化线程数量
    for i in range(len(video_page_url)):
        executor.submit(downmp4, video_page_url[i], video_name[i]).add_done_callback(call_back)  # 提交任务
    executor.shutdown(True)  # 关闭线程
    finish_time = time.time()
    print('爬虫结束，一共要爬取了%d个视频.......' % len(video_page_url))
    print("多线程总共下载的时间是" + str(finish_time - start_time))


if __name__ == '__main__':
    parsePage()
