from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir=r"C:\Users\DELL\PycharmProjects\research\named_entity_recognition-master\ResumeNER"):
    """读取数据"""
    assert split in ['train', 'dev', 'test','decode','input']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        #改为/r/n
        for line in f:
            if line != '\r\n':
                if line[0] == ' ' or line.strip() == 'O':
                    continue
                else:
                    word, tag = line.strip('\r\n').split()
                    word_list.append(word)
                    tag_list.append(tag)
            else:
                if word_list == []:
                    continue
                else:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

if __name__ == "__main__":
    build_corpus("train")
