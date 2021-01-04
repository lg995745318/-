import urllib.request
import re
from bs4 import BeautifulSoup
from distutils.filelist import findall
from queue import Queue, LifoQueue, PriorityQueue
import csv
from lxml import etree
import pymysql

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1;WOW64) AppleWebKit/537.36 (KHTML,like GeCKO) Chrome/45.0.2454.85 Safari/537.36 115Broswer/6.0.3',
    'Referer': 'http://ask.bitauto.com/browse/l2/',
    'Connection': 'keep-alive'}

# 读取一个页面的数据
def readit(headers, url):
    request = urllib.request.Request(url=url, headers=headers)
    page = urllib.request.urlopen(request)
    contents = page.read()
    soup = BeautifulSoup(contents, "html.parser")
    for tag in soup.find_all(class_='tit'):
        m_url = str(tag.find('a').get('href'))
        data = []
        try:
            m_problem = tag.get_text().strip()
            m_url = 'http://wenda' + m_url[5:]
            m_contents = getcontent(headers, m_url)
            data.extend([m_problem,m_url,m_contents])
            print(data)
            insert(data)
        except:
            continue
    return soup

def getcontent(headers, url):
    request = urllib.request.Request(url=url, headers=headers)
    page = urllib.request.urlopen(request)
    contentss = page.read()
    soup = BeautifulSoup(contentss, "html.parser")
    contents = ''
    for tag in soup.find_all('p',  class_='question', limit=2):
        try:
            contents = tag.get_text().strip()
        except:
            continue
    return contents


def create():
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='mysql', charset='utf8')# 连接数据库

    cursor = db.cursor()
    cursor.execute("DROP TABLE IF EXISTS BITAUTO")

    sql = """CREATE TABLE BITAUTO (
            ID INT PRIMARY KEY AUTO_INCREMENT,
            PROBLEM LONGTEXT,
            URL CHAR(255),
            CONTENT LONGTEXT)"""

    cursor.execute(sql)

    db.close()


def insert(value):
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='mysql', charset='utf8')

    cursor = db.cursor()
    sql = "INSERT INTO BITAUTO(PROBLEM,URL,CONTENT) VALUES (%s, %s,%s)"
    try:
        cursor.execute(sql, value)
        db.commit()
        print('插入数据成功')
    except:
        db.rollback()
        print("插入数据失败")
    db.close()


create()  # 创建表
for i in range(1,101):
    url = 'http://ask.bitauto.com/browse/l2/p' + str(i) + '/solved/'
    readit(headers, url)
    print(i)