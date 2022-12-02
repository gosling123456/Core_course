import requests
from bs4 import BeautifulSoup


def GetNovel():
    url = 'https://www.shicimingju.com/book/sanguoyanyi.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36'
    }
    res = requests.get(url=url, headers=headers)
    res.encoding = 'utf-8'
    res = res.text
    soup = BeautifulSoup(res, 'lxml')
    li_list = soup.select('.book-mulu > ul > li')
    for li in li_list:
        detail_title = li.a.string
        detail_url = 'https://www.shicimingju.com/' + li.a['href']
        content = requests.get(url=detail_url, headers=headers)
        content.encoding = 'utf-8'
        content = content.text
        article_soup = BeautifulSoup(content, 'lxml')
        article = article_soup.find('div', class_='chapter_content')
        article.encoding = 'utf-8'
        article = article.text
        content = ''.join(str(article).split())
        fp = open('./111.txt', 'a', encoding='utf-8')
        fp.write(detail_title + content + '\n')
        fp.close()
        print(detail_title + '\tDone！！！')


if __name__ == '__main__':
    GetNovel()
