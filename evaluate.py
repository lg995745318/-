import time
from collections import Counter
import numpy as np

from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
from evaluating import Metrics


def hmm_train_eval(train_data, test_data, word2id, tag2id):
    """训练并评估hmm模型"""
    # 训练HMM模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model, "./ckpts/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists)

    return pred_tag_lists


def crf_train_eval(train_data, test_data):

    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./ckpts/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists)

    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists)

    return pred_tag_lists

def ensemble_evaluate(hmm_pred, crf_pred, lstm_pred, lstmcrf_pred,latticelstm_pred, targets,status='train'):
    """ensemble多个模型"""
    tag2id1 = {'<start>': 0,'O': 1,'B-ATTRIBUTE': 2, 'M-ATTRIBUTE': 3, 'E-ATTRIBUTE': 4, 'B-OBJECT': 5, 'M-OBJECT': 6, 'E-OBJECT': 7,
              'B-CONDITION': 8, 'M-CONDITION': 9, 'E-CONDITION': 10, 'B-PARAMETERS': 11, 'M-PARAMETERS': 12,
              'E-PARAMETERS': 13,'S-ATTRIBUTE': 14, 'S-OBJECT': 15,'S-CONDITION': 16,'S-PARAMETERS': 17, '<end>': 18}
    length = len(tag2id1)
    transition = np.loadtxt(open(r"C:\Users\DELL\PycharmProjects\research\transition.csv", "rb"), delimiter="\t", skiprows=1)

    hmm_pred = append_start_end(hmm_pred)
    crf_pred = append_start_end(crf_pred)
    lstm_pred = append_start_end(lstm_pred)
    lstmcrf_pred = append_start_end(lstmcrf_pred)
    latticelstm_pred = append_start_end(latticelstm_pred)

    pred_score = []
    for i in zip(hmm_pred, crf_pred, lstm_pred, lstmcrf_pred, latticelstm_pred):
        t = []
        for j in zip(zip(*i)):
            score = np.zeros(length)
            for s in j[0]:
                score[tag2id1[s]] += 0.2
            t.append(score)
        pred_score.append(t)

    # 加入约束
    pred_tags = []
    previous = None
    reverse_tag2id = dict([(index, word) for (word, index) in tag2id1.items()])

    for p in pred_score:
        te = []
        for r in p:
            if previous is None:
                if np.argmax(r) != 0 and np.argmax(r) != 18:
                    te.append(reverse_tag2id[np.argmax(r)])
                    previous = np.argmax(r)
            else:
                pre = np.array(transition[previous])
                r += 0.5*pre
                if np.argmax(r) != 0 and np.argmax(r) != 18:
                    te.append(reverse_tag2id[np.argmax(r)])
                    previous = np.argmax(r)
        pred_tags.append(te)

    assert len(pred_tags) == len(targets)

    metrics = Metrics(targets, pred_tags)

    return pred_tags


def append_start_end(list):
    new_list = []
    for i in list:
        i.insert(0, '<start>')
        i.append('<end>')
        new_list.append(i)
    return new_list
