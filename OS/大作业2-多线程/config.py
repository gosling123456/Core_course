# -*- coding: utf-8 -*-
"""
Author  : Lamadog
Time    : 2019/12/11
Email   : xwen.xi@icloud.com
"""


url = 'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&word={}&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=111111'
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
headers = {"User-Agent": user_agent}

# -----------匹配地址------------
url2 = 'http://img\d.imgtn.bdimg.com/it/u={10},\d{10}&fm=26&gp=0.jpg'
url1 = 'https://ss\d.bdstatic.com/70cF\S\SSh_Q1YnxGkpoWK1HF6hhy/it/u=\d{10},\d{10}&fm=26&gp=0.jpg'