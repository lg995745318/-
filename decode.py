from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from ensemble_data import build_corpus
from evaluate import ensemble_evaluate
import sys
sys.path.append(r'C:\Users\DELL\PycharmProjects\research\LatticeLSTM-master')
from main import run

HMM_MODEL_PATH = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master/ckpts/hmm.pkl'
CRF_MODEL_PATH = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master/ckpts/crf.pkl'
BiLSTM_MODEL_PATH = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master/ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master/ckpts/bilstm_crf.pkl'

def decode(input_status,output_file):
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id1, tag2id = build_corpus("train", make_vocab=True)
    decode_word_lists, decode_tag_lists, word2id2, tag2id1 = build_corpus(input_status, make_vocab=True)

    lists = train_word_lists + decode_word_lists
    word2id = build_map(lists)

    print("加载并评估hmm模型...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(decode_word_lists,word2id,tag2id)

    print("加载并评估crf模型...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(decode_word_lists)

    print("加载并评估bilstm模型...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(decode_word_lists, decode_tag_lists,bilstm_word2id, bilstm_tag2id)

    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    decode_word_lists, decode_tag_lists = prepocess_data_for_lstmcrf(decode_word_lists, decode_tag_lists, test=True)
    lstmcrf_pred, target_tag_list = bilstm_model.test(decode_word_lists, decode_tag_lists,crf_word2id, crf_tag2id)

    print("加载并评估lattice lstm模型...")
    latticelstm_pred = run(status='decode')

    print("加载并评估ensemble模型...")
    predict_results = ensemble_evaluate(hmm_pred, crf_pred, lstm_pred, lstmcrf_pred, latticelstm_pred,decode_tag_lists,status='decode')

    print("输出解码结果...")
    write_decoded_results(output_file, predict_results, decode_word_lists)

def write_decoded_results(output_file, predict_results, decode_word_lists):
    fout = open(output_file, 'w', encoding='utf-8')
    sent_num = len(predict_results)
    assert (sent_num == len(decode_word_lists))
    for idx in range(sent_num):
        sent_length = len(predict_results[idx])
        for idy in range(sent_length):
            fout.write(decode_word_lists[idx][idy] + " " + predict_results[idx][idy] + '\n')
        fout.write('\n')
    fout.close()
    print(("Predict result has been written into file. %s" % (output_file)))

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

if __name__ == '__main__':
    output_file = r'C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master\ResumeNER\decode_result.char.bmes'
    input_status = 'decode'
    decode(input_status,output_file)