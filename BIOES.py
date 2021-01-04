import json
import csv
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)
from os.path import join
# import synonyms
import pymysql

def get_mark_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        j = json.load(f)['outputs']['annotation']['T']
        mark_dict = {}
        for i in range(len(j)):
            try:
                if j[i]['value'] not in mark_dict.keys():
                    mark_dict[j[i]['value']] = j[i]['name']
            except:
                pass
    return mark_dict

def get_dict_all():
    with open(r'C:\Users\DELL\PycharmProjects\research\LatticeLSTM-master\data\output_decode.char','r',encoding='utf-8') as o:
        CONDITION = {}
        OBJECT = {}
        ATTRIBUTE = {}
        PARAMETERS = {}
        conditionid = 10**10
        objectid = 10**11
        attributeid = 10**12
        parametersid = 10**13
        o_list = o.readlines()
        for index,line in enumerate(o_list):
            content = ''
            if line == '\n':
                continue
            elif line[2] == 'O':
                continue
            elif line[2] == 'S':
                if line[2:] == 'S-CONDITION\n':
                    if line[0] not in CONDITION.values():
                        conditionid += 1
                        CONDITION[conditionid] = line[0]
                if line[2:] == 'S-OBJECT\n':
                    if line[0] not in OBJECT.values():
                        objectid += 1
                        OBJECT[objectid] = line[0]
                if line[2:] == 'S-ATTRIBUTE\n':
                    if line[0] not in ATTRIBUTE.values():
                        attributeid += 1
                        ATTRIBUTE[attributeid] = line[0]
                if line[2:] == 'S-PARAMETERS\n':
                    if line[0] not in PARAMETERS.values():
                        parametersid += 1
                        PARAMETERS[parametersid] = line[0]
            elif line[2:] == 'B-CONDITION\n':
                content += line[0]
                for i in range(index+1,index+100):
                    if o_list[i][2:] == 'I-CONDITION\n':
                        content += o_list[i][0]
                    elif o_list[i][2:] == 'E-CONDITION\n':
                        content += o_list[i][0]
                        break
                if content not in CONDITION.values():
                    conditionid += 1
                    CONDITION[conditionid] = content
            elif line[2:] == 'B-OBJECT\n':
                content += line[0]
                for i in range(index+1,index+100):
                    if o_list[i][2:] == 'I-OBJECT\n':
                        content += o_list[i][0]
                    elif o_list[i][2:] == 'E-OBJECT\n':
                        content += o_list[i][0]
                        break
                if content not in OBJECT.values():
                    objectid += 1
                    OBJECT[objectid] = content
            elif line[2:] == 'B-ATTRIBUTE\n':
                content += line[0]
                for i in range(index+1,index+100):
                    if o_list[i][2:] == 'I-ATTRIBUTE\n':
                        content += o_list[i][0]
                    elif o_list[i][2:] == 'E-ATTRIBUTE\n':
                        content += o_list[i][0]
                        break
                if content not in ATTRIBUTE.values():
                    attributeid += 1
                    ATTRIBUTE[attributeid] = content
            elif line[2:] == 'B-PARAMETERS\n':
                content += line[0]
                for i in range(index+1,index+100):
                    if o_list[i][2:] == 'I-PARAMETERS\n':
                        content += o_list[i][0]
                    elif o_list[i][2:] == 'E-PARAMETERS\n':
                        content += o_list[i][0]
                        break
                if content not in PARAMETERS.values():
                    parametersid += 1
                    PARAMETERS[parametersid] = content
    return CONDITION,OBJECT,ATTRIBUTE,PARAMETERS

def json_to_txt_bioes(file):
    mark_dict = get_mark_dict(file)
    with open('autohome_bioes.txt', 'w', encoding='utf-8') as fb:
        with open('question_simple.txt', 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                result = mark_sentence(line, mark_dict)
                for i in range(len(result)):
                    for key in result[i].keys():
                        if result[i][key] != 'O':
                            if len(key) == 1:
                                res = key + ' S-' + result[i][key] + '\n'
                            elif len(key) == 2:
                                res = key[0] + ' B-' + result[i][key] + '\n'
                                res += key[-1] + ' E-' + result[i][key] + '\n'
                            elif len(key) >= 3:
                                res = key[0] + ' B-' + result[i][key] + '\n'
                                for k in range(len(key)):
                                    if k != 0 and k != len(key)-1 :
                                        res += key[k] + ' I-' + result[i][key] + '\n'
                                res += key[-1] + ' E-' + result[i][key] + '\n'
                        else:
                            res = ''
                            for o in range(len(key)):
                                if key[o] != ' ':
                                    res += key[o] + ' O' + '\n'
                        fb.writelines(res)

def automobile_bioes():
    with open('automobile.txt', 'r', encoding='utf-8') as f:
        with open('auto-bioes.txt', 'w', encoding='utf-8') as g:
            for line in f.readlines():
                line = line.replace('\n', '')
                n = len(line)
                for i in range(n):
                    if i == 0:
                        g.writelines(line[0] + ' B-ATTRIBUTE' + '\n')
                    elif i == n - 1:
                        g.writelines(line[n - 1] + ' E-ATTRIBUTE' + '\n\n')
                    else:
                        g.writelines(line[i] + ' I-ATTRIBUTE' + '\n')
    with open('NER/autohome_bioes.txt', 'r', encoding='utf-8') as f:
        with open('NER/auto-bioes.txt', 'r', encoding='utf-8') as g:
            with open('NER/signed_bioes.txt', 'w', encoding='utf-8') as s:
                for line1 in g.readlines():
                    s.writelines(line1)
                for line2 in f.readlines():
                    s.writelines(line2)

def mark_sentence(s, mark_dict):
    if s:
        for key in mark_dict.keys():
            result = s.split(key,1)
            if len(result) > 1:
                return mark_sentence(result[0], mark_dict) + [{key:mark_dict[key]}] + mark_sentence(result[1], mark_dict)
        return [{s:'O'}]
    else:
        return []

def train_dev_test():
    with open('signed_bioes.txt', 'r', encoding='utf-8') as fb:
        with open('signed_train.txt', 'w', encoding='utf-8') as ft:
            with open('signed_dev.txt', 'w', encoding='utf-8') as fd:
                with open('signed_test.txt', 'w', encoding='utf-8') as fs:
                    count = 0
                    for line in fb.readlines():
                        if line == '\n':
                            count += 1
                        if count <= 500:
                            ft.writelines(line)
                        elif count > 500 and count <= 600:
                            fs.writelines(line)
                        elif count > 600:
                            fd.writelines(line)

def get_decode_txt():
    with open('question.txt','r',encoding='utf-8') as q:
        with open('question_decode.txt','w',encoding='utf-8') as d:
            for line in q.readlines():
                for i in line:
                    if i != '\n':
                        i += ' O' + '\n'
                        d.writelines(i)
                    else:
                        d.writelines(i)

def entity_csv(file):
    CONDITION, OBJECT, ATTRIBUTE, PARAMETERS = get_dict_all()
    QUESTION = {}
    questionid = 10**9
    with open('question.txt', 'r', encoding='utf-8') as fpr:
        for value in fpr.readlines():
            questionid += 1
            QUESTION[questionid] = value.rstrip()

    active = False
    if active:
        with open('question.csv', 'w', encoding='utf-8-sig', errors='ignore') as fq:
            [fq.write('{0},{1}\n'.format('questionid', 'question'))]
            [fq.write('{0},{1}\n'.format(key, value)) for key,value in QUESTION.items()]
        with open('condition.csv', 'w',encoding='utf-8-sig') as fc:
            [fc.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format('conditionid', 'condition','syno1','syno1_sim','syno2','syno2_sim','syno3','syno3_sim'))]
            for key, value in CONDITION.items():
                near = synonyms.nearby2(value)
                [fc.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(key, value,near[0],near[1],near[2],near[3],near[4],near[5]))]
        with open('object.csv', 'w',encoding='utf-8-sig') as fo:
            [fo.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format('objectid', 'object','syno1','syno1_sim','syno2','syno2_sim','syno3','syno3_sim'))]
            for key, value in OBJECT.items():
                near = synonyms.nearby2(value)
                [fo.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(key, value,near[0],near[1],near[2],near[3],near[4],near[5]))]
        with open('attribute.csv', 'w',encoding='utf-8-sig') as fa:
            [fa.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format('attributeid', 'attribute','syno1','syno1_sim','syno2','syno2_sim','syno3','syno3_sim'))]
            for key, value in ATTRIBUTE.items():
                near = synonyms.nearby2(value)
                [fa.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(key, value,near[0],near[1],near[2],near[3],near[4],near[5]))]
        with open('parameters.csv', 'w',encoding='utf-8-sig') as fp:
            [fp.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format('parametersid', 'parameters','syno1','syno1_sim','syno2','syno2_sim','syno3','syno3_sim'))]
            for key, value in PARAMETERS.items():
                near = synonyms.nearby2(value)
                [fp.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(key, value,near[0],near[1],near[2],near[3],near[4],near[5]))]
    return QUESTION,CONDITION,OBJECT,ATTRIBUTE,PARAMETERS

def relation_csv(file):
    CONDITION, OBJECT, ATTRIBUTE, PARAMETERS = get_dict_all()
    QUESTION = {}
    questionid = 10 ** 9
    with open('question.txt', 'r', encoding='utf-8') as fpr:
        for value in fpr.readlines():
            questionid += 1
            QUESTION[questionid] = value.rstrip()

    code_dict = {**QUESTION, **CONDITION, **OBJECT, **ATTRIBUTE, **PARAMETERS}
    reverse_code_dict = {v: k for k, v in code_dict.items()}

    mark_dict = {}
    for i in CONDITION.values():
        mark_dict[i] = 'CONDITION'
    for i in OBJECT.values():
        mark_dict[i] = 'OBJECT'
    for i in ATTRIBUTE.values():
        mark_dict[i] = 'ATTRIBUTE'
    for i in PARAMETERS.values():
        mark_dict[i] = 'PARAMETERS'

    with open('question.txt', 'r', encoding='utf-8') as fp:
        with open('condition_question1.csv','w',encoding='utf-8') as cq:
            with open('object_question1.csv', 'w', encoding='utf-8') as oq:
                with open('attribute_question1.csv', 'w', encoding='utf-8') as aq:
                    with open('parameters_question1.csv', 'w', encoding='utf-8') as pq:
                        [cq.write('{0},{1}\n'.format('questionid', 'conditionid'))]
                        [oq.write('{0},{1}\n'.format('questionid', 'objectid'))]
                        [aq.write('{0},{1}\n'.format('questionid', 'attributeid'))]
                        [pq.write('{0},{1}\n'.format('questionid', 'parametersid'))]
                        for line in fp:
                            questionid = reverse_code_dict[line.rstrip()]
                            result = mark_sentence(line, mark_dict)
                            print(result)
                            for i in range(len(result)):
                                for key,value in result[i].items():
                                    if value != 'O':
                                        id = reverse_code_dict[key]
                                        if len(str(id)) == 11:
                                            [cq.write('{0},{1}\n'.format(questionid, id))]
                                        elif len(str(id)) == 12:
                                            [oq.write('{0},{1}\n'.format(questionid, id))]
                                        elif len(str(id)) == 13:
                                            [aq.write('{0},{1}\n'.format(questionid, id))]
                                        elif len(str(id)) == 14:
                                            [pq.write('{0},{1}\n'.format(questionid, id))]

def entity_count():
    CONDITION, OBJECT, ATTRIBUTE, PARAMETERS = get_dict_all()
    code_dict = {**CONDITION, **OBJECT, **ATTRIBUTE, **PARAMETERS}
    reverse_code_dict = {v: k for k, v in code_dict.items()}
    with open('question.txt', 'r', encoding='utf-8') as fp:
        for index,line in enumerate(fp.readlines()):
            s = mark_sentence(line,reverse_code_dict)
            entity = []
            count = 0
            for i in s:
                for h in i.keys():
                    if (h not in entity) and (h in code_dict.values()):
                        entity.append(h)
                        count += 1

            db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='paperdata',charset='utf8')
            cursor = db.cursor()
            sql = f"INSERT INTO QUESTIONDATA (PROBLEM,URL,QUESTION_DESCRIPTION,BROWSE_RECORD,DATE_TIME,ANSWER1,LIKE_COUNT1," \
                  f"RESPONDENTS_FANS1,ANSWER2,LIKE_COUNT2,RESPONDENTS_FANS2,ENTITY_COUNT) " \
                  f"SELECT PROBLEM,URL,QUESTION_DESCRIPTION,BROWSE_RECORD,DATE_TIME,ANSWER1,LIKE_COUNT1," \
                  f"RESPONDENTS_FANS1,ANSWER2,LIKE_COUNT2,RESPONDENTS_FANS2,{count} FROM AUTOHOME WHERE PROBLEM = %s LIMIT 1"
            try:
                cursor.execute(sql,line[:line.index('————————：')])
                db.commit()
                print("插入成功")
            except:
                db.rollback()
                print("插入失败")
            db.close()

def get_anwser(problem):
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='paperdata', charset='utf8')
    cursor = db.cursor()
    sql = f"SELECT CONTENT FROM BITAUTO WHERE PROBLEM = %s"
    try:
        cursor.execute(sql,problem)
        db.commit()
        anwser = cursor.fetchone()
        print('获取答案成功',anwser)
        return anwser
    except:
        db.rollback()
        print("获取答案失败")
    db.close()
    return ''

def update_question():
    fq = csv.reader(open(r'C:\Users\DELL\PycharmProjects\research\NER/question.csv', 'r', encoding='utf-8-sig'))
    data = [('questionid', 'question', 'browse_record', 'date_time', 'answer1', 'like_count1',
             'respondents_fans1', 'answer2', 'like_count2', 'respondents_fans2', 'entity_count')]
    sql = f"SELECT BROWSE_RECORD,DATE_TIME,ANSWER1,LIKE_COUNT1,RESPONDENTS_FANS1," \
          f"ANSWER2,LIKE_COUNT2,RESPONDENTS_FANS2,ENTITY_COUNT FROM QUESTIONDATA WHERE PROBLEM = %s LIMIT 1"
    for li in fq:
        line = ''.join(li[1:])
        try:
            db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='paperdata',
                                 charset='utf8')
            cursor = db.cursor()
            cursor.execute(sql, line[:line.index('————————：')])
            db.commit()
            result = cursor.fetchone()
            data.append((li[0], line, result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                         result[7], result[8]))
        except:
            result = ['', '', '', '', '', '', '', '', '']
            data.append((li[0], line, result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                         result[7], result[8]))
            db.rollback()
            print("失败")
    db.close()

    f = open(r'C:\Users\DELL\PycharmProjects\research\NER/question_all.csv', 'w', encoding='utf-8-sig', newline='')
    writer = csv.writer(f)
    for i in data:
        writer.writerow(i)
    f.close()

    f = open(r'C:\Users\DELL\PycharmProjects\research\NER/question_all.csv', 'w', encoding='utf-8-sig', newline='')
    writer = csv.writer(f)
    for i in data:
        writer.writerow(i)
    f.close()

def prepare_my_data():
    splits = ['train', 'dev', 'test']
    data_dir = "./ResumeNER"
    for split in splits:
        with open(join(data_dir, split + ".txt"), 'r', encoding='utf-8') as f:
            with open(join(data_dir, split + ".char.bmes"), 'w', encoding='utf-8') as d:
                file = f.readlines()
                n = len(file)
                for i  in range(n):
                    if file[i].strip() == '' and file[max(0,i-1)].strip() == '':
                        continue
                    elif file[i][1] == 'O':
                        continue
                    elif file[i].strip() == '' and file[max(0,i-1)].strip() != '':
                        d.write(file[i])
                    elif file[i][2] == 'I':
                        new_line = file[i][:2] + 'M' + file[i][3:]
                        d.write(new_line)
                    else:
                        d.write(file[i])

'''
# neo4j建立节点
LOAD CSV WITH HEADERS FROM 'file:///question_all.csv' AS line
CREATE (q:Question {questionid:toInteger(line.questionid),name:line.question,browse_record:toInteger(line.browse_record),
date_time:line.date_time,answer1:line.answer1,like_count1:toInteger(line.like_count1),respondents_fans1:toInteger(line.respondents_fans1),
answer2:line.answer2,like_count2:toInteger(line.like_count2),respondents_fans2:toInteger(line.respondents_fans2),entity_count:toInteger(line.entity_count)})

LOAD CSV WITH HEADERS FROM 'file:///attribute.csv' AS line
CREATE (a:Attribute {attributeid:toInteger(line.attributeid),name:line.attribute,
syno1:line.syno1,syno1_sim:line.syno1_sim,syno2:line.syno2,syno2_sim:line.syno2_sim,syno3:line.syno3,syno3_sim:line.syno3_sim})

LOAD CSV WITH HEADERS FROM 'file:///condition.csv' AS line
CREATE (c:Condition {conditionid:toInteger(line.conditionid),name:line.condition,
syno1:line.syno1,syno1_sim:line.syno1_sim,syno2:line.syno2,syno2_sim:line.syno2_sim,syno3:line.syno3,syno3_sim:line.syno3_sim})

LOAD CSV WITH HEADERS FROM 'file:///object.csv' AS line
CREATE (o:Object {objectid:toInteger(line.objectid),name:line.object,
syno1:line.syno1,syno1_sim:line.syno1_sim,syno2:line.syno2,syno2_sim:line.syno2_sim,syno3:line.syno3,syno3_sim:line.syno3_sim})

LOAD CSV WITH HEADERS FROM 'file:///parameters.csv' AS line
CREATE (p:Parameters {parametersid:toInteger(line.parametersid),name:line.parameters,
syno1:line.syno1,syno1_sim:line.syno1_sim,syno2:line.syno2,syno2_sim:line.syno2_sim,syno3:line.syno3,syno3_sim:line.syno3_sim})

# neo4j建立关系
LOAD CSV WITH HEADERS FROM "file:///attribute_question.csv" AS line 
match (from:Attribute{attributeid:toInteger(line.attributeid)}),(to:Question{questionid:toInteger(line.questionid)})  
merge (from)-[r:attributeof{attributeid:toInteger(line.attributeid),questionid:toInteger(line.questionid)}]->(to)

LOAD CSV WITH HEADERS FROM "file:///condition_question.csv" AS line 
match (from:Condition{conditionid:toInteger(line.conditionid)}),(to:Question{questionid:toInteger(line.questionid)})  
merge (from)-[r:conditionof{conditionid:toInteger(line.conditionid),questionid:toInteger(line.questionid)}]->(to)

LOAD CSV WITH HEADERS FROM "file:///object_question.csv" AS line 
match (from:Object{objectid:toInteger(line.objectid)}),(to:Question{questionid:toInteger(line.questionid)})  
merge (from)-[r:objectof{objectid:toInteger(line.objectid),questionid:toInteger(line.questionid)}]->(to)

LOAD CSV WITH HEADERS FROM "file:///parameters_question.csv" AS line 
match (from:Parameters{parametersid:toInteger(line.parametersid)}),(to:Question{questionid:toInteger(line.questionid)})  
merge (from)-[r:parametersof{parametersid:toInteger(line.parametersid),questionid:toInteger(line.questionid)}]->(to)
'''

if __name__ == '__main__':
    file = r"C:\Users\DELL\PycharmProjects\research\NER\outputs\outputs/question_simple.json"
    # json_to_txt_bioes(file)
    # train_dev_test()
    # entity_csv(file)
    # relation_csv(file)
    # get_decode_txt()
    # get_dict_all()
    # entity_count()
    relation_csv(file)