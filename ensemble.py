from ensemble_data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, bilstm_train_and_eval,ensemble_evaluate

import sys
sys.path.append(r'C:\Users\DELL\PycharmProjects\research\LatticeLSTM-master')
from main import run


def ensemble():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists,word2id1,tag2id = build_corpus("train",make_vocab=True)
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    decode_word_lists,decode_tag_lists,word2id2,tag2id1 = build_corpus("decode", make_vocab=True)

    lists = train_word_lists + decode_word_lists
    word2id = build_map(lists)


    print("正在训练评估Lattice LSTM模型...")
    best_acc, best_p, best_r, best_test, latticelstm_pred = run(status='train')

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )

    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )

    # 训练评估BI-LSTM模型
    print("正在训练评估双向LSTM模型...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        crf=False
    )

    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id
    )

    print("正在训练评估集成模型...")
    ensemble_evaluate(
        hmm_pred, crf_pred, lstm_pred, lstmcrf_pred,latticelstm_pred,
        test_tag_lists
    )

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

if __name__ == "__main__":
    ensemble()
