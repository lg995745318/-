'''
接收原始问题
利用lattice LSTM提取原始问题关键信息
利用关键信息构建CQL查询语句
通过积分规则确定返回的问题（答案）
'''

import sys, os
sys.path.append(r"C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master")
from decode import decode

import argparse
import main
import torch
import _pickle as pickle
from py2neo import Graph
import pymysql
import jieba
from gensim import corpora,models,similarities
import numpy as np
from datetime import datetime
import math
import cmath
import pandas as pd
from neo4j import GraphDatabase
import csv

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class Question():
    #对用户问题进行处理并返回结果
    def question_process(self,question,count=0):
        # 接收问题
        self.raw_question=str(question).strip()
        self.raw_question += ' 必要 废话'
        with open(r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master\ResumeNER\input.char.bmes','w', encoding='utf-8') as f:
            for word in self.raw_question:
                if word.rstrip() != '':
                    word += ' O' + '\n'
                else:
                    word = '\n'
                f.writelines(word)

        self.output_file = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master\ResumeNER\output.char.bmes'
        self.output_file_lattice = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master\ResumeNER\lattice_output.char.bmes'
        decode('input',self.output_file)
        # 提取出命名实体
        ner_dict = self.get_ner()
        # 查询图数据库,得到答案
        self.answer = self.query_neo4j(ner_dict,count,question)
        return self.answer

    #获取用户问题中的实体
    def get_ner(self):
        with open(self.output_file_lattice,'r',encoding='utf-8') as fo:
            ner_dict = {}
            count = 0
            for line in fo.readlines():
                count += 1
                try:
                    if line[2] != 'O':
                        if line[2] == 'S':
                            ner_dict[line[4:].strip()] = line[0]
                        elif line[2] == 'B':
                            index = 0
                            ner = ''
                            with open(self.output_file, 'r', encoding='utf-8') as f:
                                for li in f.readlines():
                                    index += 1
                                    if index >= count:
                                        if li[2] != 'E':
                                            ner += li[0]
                                        else:
                                            ner += li[0]
                                            break
                            ner_dict[line[4:].strip()] = ner
                except:
                    pass
        print('识别到实体：',ner_dict)
        return ner_dict

    #利用相关实体查询图数据库
    def query_neo4j(self,ner_dict,count,question):
        # 依托实体词典中的实体，构造cql语句查询neo4j图数据库
        self.graph = Graph("http://localhost:7474", username="neo4j",password="123456")
        # 构造cql语句
        # try:
        answer_dict,all_data = self.run_cql(ner_dict)
        all_key = [q[0] for q in all_data]
        if max(answer_dict.values()) <= 0:
            final_answer = "我也还不知道！"
        else:
            all_anwser_list = []
            for key, value in answer_dict.items():
                if value == max(answer_dict.values()):
                    try:
                        anwser = all_data[all_key.index(key)]
                        if anwser is not None:
                            all_anwser_list.append(anwser)
                    except:
                        pass

            sim = self.get_sim(all_anwser_list,question)
            score,lenaN,entiaN,readaaN,hotN,duraaN = self.compute_score(all_anwser_list,sim)
            score_list = score.tolist()
            index = score_list.index(max(score_list))
            if index%2 == 0:
                final_answer = all_anwser_list[index//2][3]
            else:
                final_answer = all_anwser_list[(index-1)//2][6]

            #获取训练数据
            # self.get_train_data(count, question,all_anwser_list, lenaN, entiaN, readaaN, hotN, duraaN, sim)
        # except:
        #     final_answer = "我也还不知道！"
        return final_answer

    #图数据库查询语句
    def run_cql(self,ner_dict):
        weight_dict = {'CONDITION': 0.5, 'OBJECT': 0.3, 'ATTRIBUTE': 0.7, 'PARAMETERS': 0.9}
        answer_dict = {'我也还不知道!': 0}
        Attribute, Condition, Object, Parameters = self.get_all_ner()
        all_data = []
        for key, value in ner_dict.items():
            #可加入词语相似度算法，相似度大于一定阈值则匹配该查询，但需要遍历所有节点，拖慢查询速度
            w = weight_dict[key]
            key = key.capitalize()
            value_dict = {value:1}
            if key == 'Attribute':
                for value1 in Attribute:
                    similarity = self.xsd(value1,value)
                    if similarity >= 0.6:
                        value_dict[value1] = similarity
            if key == 'Condition':
                for value1 in Condition:
                    similarity = self.xsd(value1, value)
                    if similarity >= 0.6:
                        value_dict[value1] = similarity
            if key == 'Object':
                for value1 in Object:
                    similarity = self.xsd(value1, value)
                    if similarity >= 0.6:
                        value_dict[value1] = similarity
            if key == 'Parameters':
                for value1 in Parameters:
                    similarity = self.xsd(value1, value)
                    if similarity >= 0.6:
                        value_dict[value1] = similarity
            print('扩展实体为：',value_dict)
            for sim_value,simi in value_dict.items():
                cql = f"match (c:{key})-[r]->(q:Question) " \
                      f"where c.name='{sim_value}' OR c.syno1='{sim_value}' OR c.syno2='{sim_value}' OR c.syno3='{sim_value}' " \
                      f"OR c.entity1='{sim_value}' OR c.entity2='{sim_value}' " \
                      f"return " \
                      f"CASE " \
                      f"WHEN c.name ='{sim_value}' THEN [q.name,q.browse_record,q.date_time,q.answer1,q.like_count1,q.respondents_fans1,q.answer2,q.like_count2,q.respondents_fans2,q.entity_count,'1','{simi}'] " \
                      f"WHEN c.syno1 ='{sim_value}' THEN [q.name,q.browse_record,q.date_time,q.answer1,q.like_count1,q.respondents_fans1,q.answer2,q.like_count2,q.respondents_fans2,q.entity_count,c.syno1_sim,'{simi}'] " \
                      f"WHEN c.syno2 ='{sim_value}' THEN [q.name,q.browse_record,q.date_time,q.answer1,q.like_count1,q.respondents_fans1,q.answer2,q.like_count2,q.respondents_fans2,q.entity_count,c.syno2_sim,'{simi}'] " \
                      f"WHEN c.syno3 ='{sim_value}' THEN [q.name,q.browse_record,q.date_time,q.answer1,q.like_count1,q.respondents_fans1,q.answer2,q.like_count2,q.respondents_fans2,q.entity_count,c.syno3_sim,'{simi}'] " \
                      f"ELSE 0 END AS RESULT"
                answer = self.graph.run(cql).data()
                answer_list = [i['RESULT'] for i in answer]
                all_data += answer_list
                for q in answer_list:
                    if q == 0:
                        continue
                    if q[0] not in answer_dict.keys():
                        answer_dict[q[0]] = w*float(q[-1])*float(q[-2])
                    else:
                        answer_dict[q[0]] += w*float(q[-1])*float(q[-2])
        return answer_dict,all_data

    #计算答案文本与用户查询问题的相似度
    def get_sim(self,all_anwser_list,question):
        anwser_list = []
        for doc in all_anwser_list:
            anwser1 = str(doc[3]) + '的'
            cut_list1 = [word for word in jieba.cut(anwser1)]
            anwser_list.append(cut_list1)
            anwser2 = str(doc[6]) + '的'
            cut_list2 = [word for word in jieba.cut(anwser2)]
            anwser_list.append(cut_list2)
        stopwords = self.stopwordslist()

        doc_list = []
        for sentence in anwser_list:
            l = []
            for word in sentence:
                if (word not in stopwords) and (word != '\t'):
                    l.append(word)
            doc_list.append(l)

        question_list = [word for word in jieba.cut(question)]
        doc_question_list = []
        for word in question_list:
            if (word not in stopwords) and (word != '\t'):
                doc_question_list.append(word)

        dictionary = corpora.Dictionary(doc_list)
        corpus = [dictionary.doc2bow(doc) for doc in doc_list]
        doc_question_vec = dictionary.doc2bow(doc_question_list)
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        sim = index[tfidf[doc_question_vec]]
        return sim

    #计算答案文本的分数
    def compute_score(self,all_anwser_list,sim):
        n = len(all_anwser_list)
        now = datetime.strptime('2020/10/20 11:58', '%Y/%m/%d %H:%M')
        char = ['。', '？', '！']
        i = 0
        weight = [-0.07801136,1.15866212,0.37536103,0.15349801]
        interception = 0.15849968344182674
        lena = np.zeros((2*n))
        entia = np.zeros((2*n))
        lika = np.zeros((2*n))
        duraa = np.zeros((2*n))
        broNa = np.zeros((2*n))
        resFana = np.zeros((2*n))
        readaa = np.zeros((2*n))
        score = np.zeros((2*n))
        for ans in all_anwser_list:
            if ans[1] is not None:
                i += 2
                lena[i-2] = float(len(ans[3]))
                lika[i-2] = float(ans[4])
                resFana[i-2] = float(ans[5])
                entia[i-2] = float(ans[-3])
                entia[i-1] = float(ans[-3])
                tim = datetime.strptime(str(ans[2]), '%Y/%m/%d %H:%M')
                duration = (tim - now).total_seconds()
                duraa[i-2] = float(duration/3600)+0.01
                duraa[i-1] = float(duration/3600)+0.01
                broNa[i-2] = float(ans[1])
                broNa[i-1] = float(ans[1])
                if ans[6] is not None:
                    lena[i-1] = float(len(ans[6]))
                    lika[i-1] = float(ans[7])
                    resFana[i-1] = float(ans[8])
                else:
                    lena[i-1] = 0.01
                    lika[i-1] = 0.01
                    resFana[i-1] = 0.01
                #计算可读性指数
                senCa1 = 1
                senCa2 = 1
                for pp in char:
                    a1 = ans[3].split(pp)
                    senCa1 += len(a1) - 1
                readaa[i-2] = 30*((len(ans[3])-senCa1)/senCa1) - ((len(ans[3])-senCa1)/senCa1)**2
                if ans[6] is not None:
                    for pp in char:
                        a2 = ans[6].split(pp)
                        senCa2 += len(a2) - 1
                    readaa[i-1] = 30*((len(ans[6])-senCa2)/senCa2) - ((len(ans[6])-senCa2)/senCa2)**2
                else:
                    readaa[i-1] = 0.01
            else:
                lena[i - 2] = 0.01
                lika[i - 2] = 0.01
                resFana[i - 2] = 0.01
                entia[i - 2] = 0.01
                entia[i - 1] = 0.01
                duraa[i - 2] = 0.01
                duraa[i - 1] = 0.01
                broNa[i - 2] = 0.01
                broNa[i - 1] = 0.01
                lena[i - 1] = 0.01
                lika[i - 1] = 0.01
                resFana[i - 1] = 0.01
                readaa[i - 2] = 0.01
                readaa[i - 1] = 0.01

        lenaN = self.Normalization([math.log(lena[i]+1) for i in range(2*n)])
        entiaN = self.Normalization([100-(entia[i]-2.5)**2 for i in range(2*n)])
        readaaN = self.Normalization(readaa)
        hotN = self.Normalization([(math.log(resFana[i]+1))*(10+lika[i])*broNa[i] for i in range(2*n)])
        duraaN = self.Normalization([math.exp(1/(duraa[i]+0.01)) for i in range(2*n)])

        for i in range(2*n):
            score[i] = weight[2]*lenaN[i]*entiaN[i] + weight[0]*readaaN[i] + weight[1]*sim[i] + weight[3]*hotN[i]*duraaN[i] + interception

        return score,lenaN,entiaN,readaaN,hotN,duraaN

    #将数据进行最大最小标准化
    def Normalization(self,x):
        return [(float(i) - min(x)+0.01) / float(max(x) - min(x)+0.01) for i in x]

    #计算相似度前需先去掉停用词
    def stopwordslist(self):
        with open(r'C:\Users\DELL\PycharmProjects\research\stopword.txt', encoding='UTF-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return stopwords

    #输入所有构建的问题，获取返回到的问题
    def get_question_anwser(self):
        with open(r'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题/问题列表.csv', 'r', encoding='gbk') as f:
            count = 100
            for line in f.readlines():
                if count != 0:
                    question = line[2:].strip()
                    self.question_process(question, count)
                count += 1

    #获取返回到的问题的相关数据，以用来计算得分
    def get_train_data(self,count,question,all_anwser_list,lenaN,entiaN,readaaN,hotN,duraaN,sim):
        answer_list = []
        for an in all_anwser_list:
            answer_list.append(an[6])
            answer_list.append(an[9])
        all_anwser_list2 = []
        for i in all_anwser_list:
            all_anwser_list2.append(i)
            all_anwser_list2.append(i)
        data = {question: answer_list, 'all_data': all_anwser_list2,
                'lenaN': lenaN, 'entiaN': entiaN,
                'readaaN': readaaN, 'hotN': hotN, 'duraaN': duraaN, 'sim': sim}
        name = [question, 'all_data', 'lenaN', 'entiaN', 'readaaN', 'hotN', 'duraaN', 'sim']
        csvdata = pd.DataFrame(columns=name, data=data)
        csvdata.to_csv(fr'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data/{count}.csv',
                       encoding='utf_8_sig')

    #获取所有节点信息
    def get_all_ner(self):
        with open(r'C:\Users\DELL\PycharmProjects\research\NER\attribute.csv','r',encoding='utf_8_sig') as fa:
            attribute = []
            for line in csv.reader(fa):
                attribute.append(line[1])
        with open(r'C:\Users\DELL\PycharmProjects\research\NER\object.csv','r',encoding='utf_8_sig') as fo:
            object = []
            for line in csv.reader(fo):
                object.append(line[1])
        with open(r'C:\Users\DELL\PycharmProjects\research\NER\condition.csv','r',encoding='utf_8_sig') as fc:
            condition = []
            for line in csv.reader(fc):
                condition.append(line[1])
        with open(r'C:\Users\DELL\PycharmProjects\research\NER\parameters.csv','r',encoding='utf_8_sig') as fp:
            parameters = []
            for line in csv.reader(fp):
                parameters.append(line[1])
        return attribute,condition,object,parameters

    #两个词语的相似度算法
    def xsd(self, value1,value2):
        i = len(value1)
        j = len(value2)
        sz = np.zeros((i + 1, j + 1))
        for a in range(i + 1):
            sz[a][0] = a
        for a in range(j + 1):
            sz[0][a] = a
        for b in range(1, i + 1):
            for c in range(1, j + 1):
                if value1[b - 1] == value2[c - 1]:
                    temp = 0
                else:
                    temp = 1
                sz[b][c] = min(sz[b - 1][c - 1] + temp, sz[b][c - 1] + 1, sz[b - 1][c] + 1)
        similarity = 1 - (sz[i][j] / max(i, j))
        return similarity

if __name__ == '__main__':
    que = Question()
    with open(r'C:\Users\DELL\PycharmProjects\research\测试问句.csv','r',encoding='gbk') as f:
        reader = csv.reader(f)
        data = {}
        for question in reader:
            q = question[0].strip()
            a = que.question_process(q)
            data[q] = a
    with open(r'C:\Users\DELL\PycharmProjects\research\测试答案.csv', 'w', newline='',encoding='utf_8_sig') as newfile:
        filewriter = csv.writer(newfile)
        filewriter.writerow(['问题', '答案'])
        for k,v in data.items():
            filewriter.writerow([k, v])

    # question = '转向灯异常闪烁是由于什么原因'
    # anwser = que.question_process(question)
    # print(anwser)