import jieba
import pymysql
import jieba.posseg as pseg

def sql_txt(problem_out):
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='mysql', charset='utf8')

    cursor = db.cursor()
    sql = "SELECT ID,PROBLEM FROM BITAUTO"
    try:
        cursor.execute(sql)
        data = cursor.fetchall()
        fp = open('problem_out.txt', "w")
        problem_count = 0
        for problem in data:
            try:
                problem = str(problem[1]) + '\n'
                problem_count += 1
                fp.write(problem)
            except:
                pass
        fp.close()
        cursor.close()
        print("写入完成,共写入%d条数据！" % problem_count)
        db.commit()
        print('插入数据成功')
    except:
        db.rollback()
        print("插入数据失败")
    db.close()

def jieba_seg(problem_out,filename,stopwords_list,userdict):
    fn = open('problem_out.txt', "r")
    f = open(filename, "w+")
    fu = open(userdict, 'w+', encoding = 'utf-8')
    jieba.load_userdict("./automobile.txt")
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    strip = []
    for line in fn.readlines():
        words = pseg.cut(line)
        for word,pos in words:
            word = str(word)
            pos = str(pos)
            #wp = str(word) + ' ' + str(pos)
            if str(pos) == 'n':
                strip.append(word)
                word += '\n'
                fu.write(word)

            if word in stopwords_list or pos in stop_flag:
                continue
            else:
                word += ' '
                f.write(word)
    f.close()
    fn.close()

def read_stopwords(stop_name):
    stopwords_list = []
    ifs = open(stop_name, 'r', encoding='utf-8', errors='ignore')
    for line in ifs.readlines():
        line = line.strip()
        stopwords_list.append(line)
    #print(len(stopwords_list),type(stopwords_list))
    return stopwords_list

def txt_char(problem_out):
    fr = open('problem_out.txt', "r")
    fw = open('test.char', 'w+', encoding='utf-8')
    li = ['', ' ']
    for line in fr.readlines():
        line = str(line)
        for i in range(len(line)):
            if line[i] not in li:
                if line[i] == '\n':
                    fw.write(line[i])
                else:
                    lll = line[i] + ' O\n'
                    fw.write(lll)
    fr.close()
    fw.close()


if __name__ == "__main__":
    problem_out = r'C:\Users\99574\PycharmProjects\paperdata'
    filename = 'result.txt'
    stop_name = 'stopword.txt'
    userdict = 'userdict.txt'
    # sql_txt(problem_out)
    #stopwords_list = read_stopwords(stop_name)
    #jieba_seg(problem_out,filename,stopwords_list,userdict)
    txt_char(problem_out)