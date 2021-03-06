class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags):
        self.get_ner_fmeasure(golden_tags, predict_tags)

    def get_ner_fmeasure(self,golden_tags, predict_tags):
        sent_num = len(golden_tags)
        golden_full = []
        predict_full = []
        right_full = []
        right_tag = 0
        all_tag = 0
        for idx in range(0, sent_num):
            # word_list = sentence_lists[idx]
            golden_list = golden_tags[idx]
            predict_list = predict_tags[idx]
            for idy in range(len(golden_list)):
                if golden_list[idy] == predict_list[idy]:
                    right_tag += 1
            all_tag += len(golden_list)

            gold_matrix = self.get_ner_BMES(golden_list)
            pred_matrix = self.get_ner_BMES(predict_list)

            right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
            golden_full += gold_matrix
            predict_full += pred_matrix
            right_full += right_ner
        right_num = len(right_full)
        golden_num = len(golden_full)
        predict_num = len(predict_full)
        if predict_num == 0:
            precision = -1
        else:
            precision = (right_num + 0.0) / predict_num
        if golden_num == 0:
            recall = -1
        else:
            recall = (right_num + 0.0) / golden_num
        if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
            f_measure = -1
            accuracy = 0
        else:
            f_measure = 2 * precision * recall / (precision + recall)
            accuracy = (right_tag + 0.0) / (all_tag + 0.0000000001)
        # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
        print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
        print("accuracy = ", accuracy, " precision = ", precision, " recall = ", recall," f_measure = ", f_measure)
        return accuracy, precision, recall, f_measure

    def get_ner_BMES(self,label_list):
        # list_len = len(word_list)
        # assert(list_len == len(label_list)), "word list size unmatch with label list"
        list_len = len(label_list)
        begin_label = 'B-'
        end_label = 'E-'
        single_label = 'S-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = label_list[i].upper()
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = self.reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        # print stand_matrix
        return stand_matrix

    def reverse_style(self,input_string):
        target_position = input_string.index('[')
        input_len = len(input_string)
        output_string = input_string[target_position:input_len] + input_string[0:target_position]
        return output_string
