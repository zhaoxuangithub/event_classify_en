import os
import sys
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath('..'))


def count():
    tag_dic = {}
    directory = 'D:/工作2020/wc/wiki_ch_ner'
    for fn in os.listdir(directory):
        fp = directory + os.sep + fn
        txt = []
        with open(fp, 'r', encoding='utf-8') as fr:
            for line in fr:
                txt.append(line.rstrip('\n'))
        for sen in txt:
            soup = BeautifulSoup(sen, 'html.parser')
            for token in soup.children:
                if not token.string:
                    continue
                if token.name:
                    if token.name in tag_dic.keys():
                        temp_dic = tag_dic[token.name]
                        word = token.string
                        if word in temp_dic.keys():
                            temp_dic[word] += 1
                        else:
                            temp_dic[word] = 1
                    else:
                        tag_dic[token.name] = {}
                        tag_dic[token.name][token.string] = 1
    for k in tag_dic.keys():
        print(k + ': ' + str(len(tag_dic[k])))
    # print(tag_dic)


if __name__ == '__main__':
    # count()
    s = 'svm.m'
    print(s[:-2])
