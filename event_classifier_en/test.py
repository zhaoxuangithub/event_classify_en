from sklearn.datasets.base import Bunch
import shutil
import os
import json
import jieba
import numpy
# from web_classifier.utils import json_combine
from utils import json_combine
# from fast_server import init_spacy_model
import spacy

# bunch = Bunch(target_name=[1,2,3], label=['a','b','c'], filenames=['a.t','b.t','c.t'], contents=['测试','已经','通过'])
# out = bunch.target_name[1]
# print(out).


# def _dircheck(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# s = 'a' + os.sep + 'b' + os.sep
# _dircheck(s)
# def test():
#     tup = (1,2,3)
#     t1,t2,t3 = tup
#     return t1,t2,t3
#
# a,b,c = test()
# print(b)

def gen_train_data():
    name = 'F:/工作/2月份文本分类语料/search_0228_'
    i = 2
    t_list = []
    while i < 5:
        new_name = name + str(i) + '.txt'
        with open(new_name, 'r', encoding='utf-8') as fr:
            line = fr.readline()
            while line:
                dic = {}
                data = json.loads(line)
                if not data['from'] == 'biying':
                    dic['content'] = data['content']
                    dic['class'] = data['keyword']
                    jd = json.dumps(dic, ensure_ascii=False)
                    t_list.append(jd)
                line = fr.readline()
        i += 1
    with open('F:/工作/2月份文本分类语料/train_0304.txt', 'a', encoding='utf-8') as fw:
        dic = {}
        dic['model'] = 'bayes'
        dic['model_name'] = 'bayes'
        dic['scale'] = '0.4'
        jd = json.dumps(dic, ensure_ascii=False)
        fw.write(jd)
        fw.write('\n')
        for t in t_list:
            fw.write(t)
            fw.write('\n')


def loot():
    print(os.path.abspath('..'))


def cut_test():
    s = 'Working teams of China and the US are still consulting, as there are many things for the two sides to do and joint efforts are demanded to achieve mutual consensus, Chinese Minister of Commerce Zhong Shan said in commenting about the latest progress in China-US trade talks Tuesday.'
    print ("|".join(jieba.cut(s)))


def c_t():
    with open('F:/工作/2月份文本分类语料/sgns.merge.word', 'r', encoding='utf-8') as fr:
        i = 0
        for line in fr:
            if i == 0:
                first_line = line.split(' ')
                word_num = int(first_line[0])
                word_dimension = int(first_line[1])
                i += 1
                continue
            i += 1
            dic_entry = line.split(' ')
            print(type(dic_entry[301]))
            # for p in dic_entry[1:302]:
            #     i+=1
            #     # print(p)
            #     # print(type(p))
            #     print(i)
            #     c = float(p)
            #     print(c)
            # # l = [float(c) for c in dic_entry[4:]]
            # # print(l)
            # # print(numpy.asarray(l, dtype='float32'))
            break


def list_t():
    l = [1, 2, 3]
    c = [str(i) for i in l[1:2]]
    print(c)


def convert():
    s = '-0.195045'
    print(float(s))


def tstst():
    s = 'iashdkjsabdhbn'
    p = s.split(' ')
    print(p)


def local_train():
    path = './data'
    name_list = os.listdir(path)
    t_list = []
    for filename in name_list:
        filepath = path + os.sep + filename
        result_path = ''
        with open(filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                dic = {}
                line = line.rstrip('\n')
                dic['content'] = line
                dic['class'] = filename.rstrip('.txt')
                jd = json.dumps(dic, ensure_ascii=False)
                t_list.append(jd)
    with open('D:/工作2020/train_0702.txt', 'w', encoding='utf-8') as fw:
        dic = {}
        dic['model'] = 'svm'
        dic['model_name'] = '0702'
        dic['scale'] = '0.2'
        jd = json.dumps(dic, ensure_ascii=False)
        fw.write(jd)
        fw.write('\n')
        for t in t_list:
            fw.write(t)
            fw.write('\n')


def new_data():
    path = 'D:/工作2020/wiki词条分类数据/'
    name_list = os.listdir(path)
    save_path = './data'
    for filename in name_list:
        filepath = path + filename
        ls = []
        with open(filepath, 'r', encoding='utf-8') as fr:
            count = 0
            for line in fr:
                count += 1
                if count > 1400:
                    break
                ls.append(json_combine(line))
        write_path = save_path + os.sep + filename
        with open(write_path, 'w', encoding='utf-8') as fw:
            for l in ls:
                fw.write(l)


def data_test():
    with open('./data/公司.txt', 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line)
            print(type(data['property']))
            break
            
            
def handle_data(pa: str, pt: str):
    """
    处理数据文件夹中数据按照指定格式到一个文件中
    :param pa: 源文件夹路径
    :param pt: 目标路径文件
    :return:
    """
    names = os.listdir(pa)
    data_list = []
    for n in names:
        temp_path = os.path.join(pa, n)
        with open(temp_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line and line.strip():
                    dic = {}
                    dic['content'] = line
                    dic['class'] = n.rstrip('.txt')
                    jd = json.dumps(dic, ensure_ascii=False)
                    data_list.append(jd)
    with open(pt, 'w', encoding='utf-8') as fw:
        dic = {}
        dic['model'] = 'svm'
        dic['model_name'] = '1022'
        dic['scale'] = '0.2'
        jd = json.dumps(dic, ensure_ascii=False)
        fw.write(jd + "\n")
        for t in data_list:
            fw.write(t + "\n")
            
            
def deal_format_tar_file(p1):
    """
    :param p1: 目标路径
    :return:
    """
    res = []
    with open(p1, 'r', encoding='utf-8') as fr:
        for line in fr:
            if line and line.strip():
                js = json.loads(line.strip())
                if "content" in js:
                    js["content"] = js["content"].strip()
                    class_ty = js["class"]
                    if ".json" in class_ty:
                        js["class"] = class_ty.replace(".json", "")
                    js_str = json.dumps(js, ensure_ascii=False) + "\n"
                    res.append(js_str)
                else:
                    res.append(line)
    with open(p1, 'w', encoding='utf-8') as fw:
        if res:
            for line in res:
                fw.write(line)
                
        
def deal_format_tar_file_class(p1):
    """
    :param p1: 目标路径
    :return:
    """
    res = []
    with open(p1, 'r', encoding='utf-8') as fr:
        for line in fr:
            if line and line.strip():
                js = json.loads(line.strip())
                if "content" in js:
                    js["content"] = js["content"].strip()
                    class_ty = js["class"]
                    temp_class = ""
                    if class_ty == "会议活动":
                        temp_class = "meet"
                    elif class_ty == "伤人":
                        temp_class = "hurt"
                    elif class_ty == "战斗":
                        temp_class = "battle"
                    elif class_ty == "投票表决":
                        temp_class = "vote"
                    elif class_ty == "抗议":
                        temp_class = "protest"
                    elif class_ty == "涉法律行为":
                        temp_class = "law"
                    elif class_ty == "组织机构变更":
                        temp_class = "change_organization"
                    elif class_ty == "职位变更":
                        temp_class = "change_position"
                    if temp_class:
                        js["class"] = temp_class
                    js_str = json.dumps(js) + "\n"
                    res.append(js_str)
                else:
                    res.append(line)
    with open(r'./data_cp/1103_8_en_class.txt', 'w', encoding='utf-8') as fw:
        if res:
            for line in res:
                fw.write(line)


if __name__ == '__main__':
    # gen_train_data()
    # local_train()
    
    # 1. 测试构建文件
    # mypath = r'D:/Pycharm_workspace/web_classifier/data_cp2'
    # pt = r'D:/Pycharm_workspace/web_classifier/data_cp/1022.txt'
    # handle_data(mypath, pt)
    
    # 2.
    # 处理构建好的文件./data_cp/1028_15.txt
    # deal_format_tar_file(r'./data_cp/1028_15.txt')
    
    # 3. 测试spacy
    # nlp_name = "en_core_sci_sm"
    # spacy_model = spacy.load(nlp_name)
    # # spacy_model = init_spacy_model()
    # text = "Blair and Bush were to hold a joint press conference at 1600 GMT Thursday."
    # doc = spacy_model(text)
    # content_seg = [token.orth_ for token in doc]
    # print(content_seg)
    # # ['Blair', 'and', 'Bush', 'were', 'to', 'hold', 'a', 'joint', 'press', 'conference', 'at', '1600', 'GMT', 'Thursday', '.']
    
    # 4. 处理数据
    deal_format_tar_file_class(r'./data_cp/1103_8_en_deal.txt')
    pass
