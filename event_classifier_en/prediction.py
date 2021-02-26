# encoding=utf-8

import os
import jieba
import pickle
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import json_combine


def predict_segment(content_list):
    seg_list = []
    for text in content_list:
        text = text.replace("\r\n", "")  # 删除换行
        text = text.replace(" ", "")  # 删除空行、多余的空格
        text_seg = jieba.cut(text)  # 为文件内容分词
        seg_list.append(' '.join(text_seg))
    return seg_list


def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


def pack_bunch(seg_list):
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.append('predict')
    i = 0
    for seg_txt in seg_list:
        bunch.contents.append(seg_txt)
        bunch.filenames.append(str(i))
        bunch.label.append('predict')
        i += 1
    return bunch


def vectorize(bunch, mode, model_name=None):
    stop_path = 'stopword' + os.sep + 'stopwords.txt'
    stpwrdlst = open(stop_path, encoding='utf-8').read().replace(r'\n', ' ').split()
    if model_name is None:
        vector_path = mode + os.sep + 'tfidf' + os.sep + 'tfdifspace.dat'
    else:
        vector_path = mode + os.sep + 'tfidf' + os.sep + model_name + '_train.dat'
    train_vector = _readbunchobj(vector_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames,
                       tdm=[], vocabulary={})
    tfidfspace.vocabulary = train_vector.vocabulary
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                 vocabulary=train_vector.vocabulary)
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    return tfidfspace


def predict(vector_space, mode, model, content_list):
    predict_model = joblib.load(mode + os.sep + 'model' + os.sep + model + '.m')
    predict_result = predict_model.predict(vector_space.tdm)
    json_data = []
    for file_name, expct_cate in zip(vector_space.filenames, predict_result):
        json_dic = {}
        json_dic['content'] = content_list[int(file_name)]
        json_dic['class'] = expct_cate
        json_data.append(json_dic)
    return json_data


# 此处传进来的model参数是**.m文件的名字（不包括后缀），在default为true的情况下，model只能为svm或者bayes。
def fast_prediction(content_list, default, model):
    seg_list = predict_segment(content_list)
    text_bunch = pack_bunch(seg_list)
    # f_dic ={}
    # for name, content in zip(text_bunch.filenames, text_bunch.contents):
    #     f_dic[name] = content

    if default:
        mode = 'default'
        vec_space = vectorize(text_bunch, mode)
    else:
        mode = 'custom'
        vec_space = vectorize(text_bunch, mode, model)
    result_data = predict(vec_space, mode, model, content_list)
    return result_data


# 此处传进来的model参数是**.m文件的名字（不包括后缀），在default为true的情况下，model只能为svm或者bayes。
def new_prediction(json_list, default, model):
    content_list = []
    for json_line in json_list:
        content_list.append(json_combine(json_line))
    seg_list = predict_segment(content_list)
    text_bunch = pack_bunch(seg_list)

    if default:
        mode = 'default'
        vec_space = vectorize(text_bunch, mode)
    else:
        mode = 'custom'
        vec_space = vectorize(text_bunch, mode, model)
    result_data = predict(vec_space, mode, model, content_list)
    return result_data


def detailed_predict(content_list):
    mode = 'custom'
    model = '0702_bayes'
    seg_list = predict_segment(content_list)
    text_bunch = pack_bunch(seg_list)
    vector_space = vectorize(text_bunch, mode, model)
    predict_model = joblib.load(mode + os.sep + 'model' + os.sep + model + '.m')
    res = predict_model.predict_log_proba(vector_space.tdm)
    print(res)
    res = predict_model.predict_proba(vector_space.tdm)
    print(res)
    jll = predict_model._joint_log_likelihood(vector_space.tdm)
    print(jll)
    args = np.argmax(jll, axis=1)
    print(args)


if __name__ == '__main__':
    # detailed_predict(['约翰·海因里希·冯·杜能（Johann Heinrich von Thünen，1783年6月24日－1850年9月22日），台湾多译为邱念[1]，香港译为范杜能 ，又有译屠能[3]，梅克伦堡经济学者。他的学说被认为是经济地理学和农业地理学的开创者。他被费尔南·布劳岱尔称为除了马克思之外十九世纪的最伟大的经济学者。'])
    # detailed_predict(['鲁迅（1881年9月25日－1936年10月19日），曾用名周樟寿，后改名周树人，曾字豫山，后改豫才，曾留学日本仙台医科专门学校（肄业）。“鲁迅”是他1918年发表《狂人日记》时所用的笔名，也是他影响最为广泛的笔名，浙江绍兴人。著名文学家、思想家、革命家、民主战士，五四新文化运动的重要参与者，中国现代文学的奠基人。毛泽东曾评价：“鲁迅的方向，就是中华民族新文化的方向。” [1-7]鲁迅一生在文学创作、文学批评、思想研究、文学史研究、翻译、美术理论引进、基础科学介绍和古籍校勘与研究等多个领域具有重大贡献。他对于五四运动以后的中国社会思想文化发展具有重大影响，蜚声世界文坛，尤其在韩国、日本思想文化领域有极其重要的地位和影响，被誉为“二十世纪东亚文化地图上占最大领土的作家”。'])
    # detailed_predict(['循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network） [1]  。对循环神经网络的研究始于二十世纪80-90年代，并在二十一世纪初发展为深度学习（deep learning）算法之一 [2]  ，其中双向循环神经网络（Bidirectional RNN, Bi-RNN）和长短期记忆网络（Long Short-Term Memory networks，LSTM）是常见的的循环神经网络 [3]  。循环神经网络具有记忆性、参数共享并且图灵完备（Turing completeness），因此在对序列的非线性特征进行学习时具有一定优势 [4]  。循环神经网络在自然语言处理（Natural Language Processing, NLP），例如语音识别、语言建模、机器翻译等领域有应用，也被用于各类时间序列预报。引入了卷积神经网络（Convoutional Neural Network,CNN）构筑的循环神经网络可以处理包含序列输入的计算机视觉问题。'])
    with open('./test/tt.txt', 'r', encoding='utf-8') as fr:
        line = fr.readline()
        line = line.rstrip('\n')
        print(line)
    result = new_prediction([line],
                   False,
                   # '0702_bayes'
                   '0702_svm'
                   )
    print(result)
