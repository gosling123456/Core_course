import requests
from lxml import etree
import threading, psutil, os
import time
from concurrent.futures import ThreadPoolExecutor
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3863.400 QQBrowser/10.8.4334.400',
}
def parsePage():
    url = "https://ibaotu.com/tupian/gongjiangjingshen/7-0-0-0-0-0-0.html?format_type=0"
    response = requests.get(url=url, headers=headers)
    html_content = response.text
    tree = etree.HTML(html_content)
    video_page_url = tree.xpath("//ul/li/div/div/a/div[1]/video/@src")
    print(video_page_url)
    video_name = tree.xpath("//ul/li/@pr-data-title")
    start_time = time.time()
    path = '包图网大国工匠视频普通下载'
    if not os.path.exists(path):
        os.mkdir(path)
    print('开始爬虫')
    for i in range(len(video_page_url)):
        video_url = "https:"+video_page_url[i]
        video_name = str(i)
        video_content = requests.get(url=video_url, headers=headers).content
        with open(path + './%s.mp4' % video_name, "wb") as f:
            f.write(video_content)
    finish_time = time.time()
    print('爬虫结束')
    print("普通下载总共下载的时间是" + str(finish_time - start_time))


if __name__ == '__main__':
    parsePage()