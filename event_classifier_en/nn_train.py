import jieba
import os
import numpy
import pickle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shutil
import spacy
import json

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

# gpu按需分配
# gpu_id = str(os.environ.get('DEVICE_ID'))
# print("********************gpu_id={}****************".format(gpu_id))
# None
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用1号显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # 设置使用1号显卡
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)


def my_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def my_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def my_f1(y_true, y_pred):
    precision = my_precision(y_true, y_pred)
    recall = my_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


    # model.compile(loss='binary_crossentropy',
    #           optimizer= "adam",
    #           metrics=[f1])


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        print(val_targ)
        print(val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return


def init_spacy_model():
    """
    初始化spacy model
    """
    # nlp_name = "en_core_sci_sm"
    nlp = spacy.load('en_core_web_sm')
    # nlp = spacy.load(nlp_name)
    # en_core_sci_sm下有模型进行加载
    return nlp


def _dircheck(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _readfile(path):
    with open(path, 'r', encoding='utf-8') as fr:
        content = fr.read()
        return content


def _savefile(savepath, content):
    with open(savepath, "w", encoding='utf-8') as fw:
        fw.write(content)


def analyze_file(path):
    """
    分析文件并做相关处理
    :param path:
    :return:
    """
    # 语料路径
    cp = 'custom' + os.sep + 'corpus'
    # 分段路径
    sp = 'custom' + os.sep + 'segment'
    if os.path.exists(cp):
        shutil.rmtree(cp)
    if os.path.exists(sp):
        shutil.rmtree(sp)
    with open(path, 'r', encoding='utf-8') as fr:
        first_line = fr.readline()
        params = json.loads(first_line)
        model = params['model']
        model_name = params['model_name']
        scale = float(params['scale'])
        line = fr.readline()
        num = 1
        while line:
            data = json.loads(line)
            content = data['content']
            classification = data['class']
            path = 'custom' + os.sep + 'corpus' + os.sep + classification
            _dircheck(path)
            file_path = path + os.sep + str(num) + '.txt'
            num += 1
            fw = open(file_path, 'w', encoding='utf-8')
            fw.write(content)
            fw.close()
            line = fr.readline()
    print('文件分类保存完成')
    return model, model_name, scale


def corpus_seg():
    """
    分词并保存
    :param scale:
    :return:
    """
    corpus_path = 'custom' + os.sep + 'corpus'
    clslist = os.listdir(corpus_path)
    for cls in clslist:
        class_path = corpus_path + os.sep + cls + os.sep
        seg_path = 'custom' + os.sep + 'segment' + os.sep
        _dircheck(seg_path)
        cls_seg_path = seg_path + cls + os.sep
        _dircheck(cls_seg_path)
        one_cls_list = os.listdir(class_path)
        for single_file in one_cls_list:
            fullname = class_path + single_file
            content = _readfile(fullname)
            content = content.replace('\r\n', '')
            content = content.strip()
            # content = content.replace(' ', '')
            # 分词
            # content_seg = jieba.cut(content)
            doc = spacy_model(content)
            # items = [([tok.text for tok in sent], sent.text) for sent in doc.sents if len(sent) >= 3]
            content_seg = [token.orth_ for token in doc]
            # print(content, content_seg)
            _savefile(cls_seg_path + single_file, " ".join(content_seg))  # 将处理后的文件保存到分词后语料目录
    print('分词完成！保存结束！')


def creare_model(model, model_name, vec_path, val_scale):
    seg_path = 'custom' + os.sep + 'segment' + os.sep
    label_list = os.listdir(seg_path)
    label_list.sort()
    label_path = 'labels' + os.sep + model_name + '_' + model + '_' + 'labels.txt'
    with open(label_path, 'w', encoding='utf-8') as fw:
        for l in label_list:
            fw.write(l)
            fw.write(' ')
    label_num = len(label_list)  # 标签的总数，后续生成标签序列和训练需要用到
    texts = []  # 语料集的文本序列
    labels = []  # 语料集的分类id序列
    for label in label_list:
        label_id = label_list.index(label)
        path = seg_path + label
        txt_list = os.listdir(path)
        for txt in txt_list:
            txt_path = path + os.sep + txt
            texts.append(_readfile(txt_path))
            labels.append(label_id)
    labels = numpy.array(labels, dtype=int)  # 为了做后续处理和训练，必须将各个列表转换为numpy的array
    token = Tokenizer()
    token.fit_on_texts(texts)
    seq = token.texts_to_sequences(texts)  # 文本索引化序列
    word_index = token.word_index  # 词索引字典
    dic_path = 'dictionary' + os.sep
    _dircheck(dic_path)
    dic_file = dic_path + model_name + '_' + model + '_' + 'wid.pkl'
    pickle.dump(word_index, open(dic_file, 'wb'))  # 保存词索引字典
    word_num = len(word_index)  # 整个字典的词数
    print('标签抽取完毕')
    print('词索引字典生成')
    
    word_dic = {}  # 词向量字典
    # with open('F:/工作/2月份文本分类语料/sgns.merge.word', 'r', encoding='utf-8') as fr:
    # word_dimension = 100
    word_dimension = 300
    with open(vec_path, 'r', encoding='utf-8') as fr:
        # i = 0
        # 50d
        # vec_arrary = json.loads(fr.readline().strip())
        for line in fr:
            # if i == 0:
            #     first_line = line.split(' ')
            #     word_dimension = int(first_line[1])
            #     i += 1
            #     continue
            # dic_entry = line.split(' ')
            # word_dic[dic_entry[0]] = numpy.asarray(dic_entry[1:word_dimension+1], dtype='float32')
            w_list = line.strip().split(" ")
            # 50d
            # word_dic[d["word"]] = numpy.asarray(d["vec"], dtype='float32')
            word_dic[w_list[0]] = numpy.asarray(w_list[1:], dtype='float32')
            # i += 1
    dic_file = dic_path + model_name + '_' + model + '_' + 'wvd.pkl'
    pickle.dump(word_dic, open(dic_file, 'wb'))  # 用户训练神经网络的时候，其实不需要保存这个字典，这条语句可删除
    print('词向量字典生成完毕')
    
    vec_matrix = numpy.zeros((word_num + 1, word_dimension))  # 索引-向量矩阵
    for word, index in word_index.items():
        if word in word_dic.keys():
            vec_matrix[index] = word_dic[word]
    print('矩阵生成完毕')
    
    labels_vec = np_utils.to_categorical(labels, label_num)
    texts_index = sequence.pad_sequences(seq, maxlen=1000, padding='post', truncating='post')
    # x_train, x_test, y_train, y_test = train_test_split(texts_index, labels_vec, test_size=0.2)  # 验证集比例
    x_train, x_test, y_train, y_test = train_test_split(texts_index, labels_vec, test_size=val_scale)  # 验证集比例
    if model == 'cnn':
        result = train_cnn(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    elif model == 'lstm':
        result = train_lstm(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    elif model == 'text_cnn':
        result = text_cnn(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    return result


def train_lstm(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension):
    model = Sequential()
    model.add(Embedding(input_dim=dim,
                        output_dim=word_dimension,
                        mask_zero=True,
                        weights=[matrix],
                        input_length=1000,
                        trainable=False))
    model.add(Bidirectional(LSTM(units=256), merge_mode='sum'))
    # model.add(Dropout(0.3))
    model.add(Dropout(0.2))
    # old
    # model.add(Dense(label_num, activation='relu'))
    model.add(Dense(label_num, activation='softmax'))
    print('编译模型。。。')
    # old
    # model.compile(loss='mse',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # 1130
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[my_precision, my_recall, my_f1])
    print('开始训练。。。')
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=50,
                        validation_data=(x_test, y_test),
                        verbose=1)
    model_file = 'neural' + os.sep + model_name + '_lstm.h5'
    model.save(model_file)
    print('评估。。。')
    # old
    # loss, acc = model.evaluate(x_test, y_test, batch_size=120)
    # 1130
    loss, precision, recall, f1_s = model.evaluate(x_test, y_test, batch_size=128)
    # loss_curve(history)
    result = {}
    # old
    # result['loss'] = loss
    # result['accuracy'] = acc
    # 1130
    result['loss'] = loss
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1_s
    return result


def train_cnn(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension):
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedding_layer = Embedding(dim,
                                word_dimension,
                                weights=[matrix],
                                input_length=1000,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(label_num, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        nb_epoch=10, batch_size=128, verbose=1)
    model_file = 'neural' + os.sep + model_name + '_cnn.h5'
    model.save(model_file)
    loss, acc = model.evaluate(x_test, y_test, batch_size=120)
    loss_curve(history)
    result = {}
    result['loss'] = loss
    result['accuracy'] = acc
    return result


def my_metric_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


def text_cnn(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension):
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedding_layer = Embedding(dim,
                                word_dimension,
                                weights=[matrix],
                                input_length=1000,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(1000 - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merged = concatenate(convs, axis=1)
    # out = Dropout(0.5)(merged)
    # TODO 1201
    out = Dropout(0.3)(merged)
    output = Dense(32, activation='relu')(out)
    # output = Dense(units=label_num, activation='sigmoid')(output)
    # TODO 1201
    output = Dense(units=label_num, activation='softmax')(output)
    model = Model(sequence_input, output)
    # old
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # custom
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[my_precision, my_recall, my_f1])
    
    # metrics = Metrics()
    # model.fit(X[train], Y[train], epochs=150, batch_size=batch_size,
    #           verbose=0, validation_data=(X[test], Y[test]),
    #           callbacks=[metrics])
    
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
    #                     nb_epoch=10, batch_size=128, verbose=1, callbacks=[metrics])
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
    #                     nb_epoch=10, batch_size=128, verbose=1)
    # train epoch=50
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        nb_epoch=50, batch_size=128, verbose=1)
    model_file = 'neural' + os.sep + model_name + '_text-cnn.h5'
    model.save(model_file)
    # custom
    # loss, precision, recall, f1_s = model.evaluate(x_test, y_test, batch_size=120)
    # TODO 1201
    loss, precision, recall, f1_s = model.evaluate(x_test, y_test, batch_size=64)
    # 绘制正确率曲线
    # loss_curve(history)
    result = {}
    # old
    # result['loss'] = loss
    # result['accuracy'] = acc
    # custom
    result['loss'] = loss
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1_s
    
    # metrics class
    # print("*********************")
    # print(metrics.val_f1s)
    # print(metrics.val_precisions)
    # print(metrics.val_recalls)
    # print("**************************")
    return result


def loss_curve(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def customize_train(path):
    # 解析文件各类别分成一条一文件保存下来
    model, model_name, scale = analyze_file(path)
    # 分词并保存
    corpus_seg()
    # 构建模型训练并返回结果
    # result = creare_model(model, model_name, 'vector_file', scale)
    # 50d
    # result = creare_model(model, model_name, './vector_file/glove.6B.50d.json', scale)
    # 100d
    result = creare_model(model, model_name, './vector_file/glove.6B.300d.txt', scale)
    json_data = json.dumps(result, ensure_ascii=False)
    return json_data


def nn_prediction(content_list, model, model_type, model_name):
    if model_name.startswith('default'):
        word_index = pickle.load(open('dictionary' + os.sep + 'word_index.pkl', 'rb'))
        label_path = 'labels' + os.sep + 'default_labels.txt'
    else:
        word_index = pickle.load(open('dictionary' + os.sep + model_name + '_' + model_type + '_' + 'wid.pkl', 'rb'))
        label_path = 'labels' + os.sep + model_name + '_' + model_type + '_labels.txt'
    with open(label_path, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        line = line.rstrip(' ')
        label_list = line.split(' ')
    seq_list = []
    for content in content_list:
        content = content.replace('\r\n', '')
        # content = content.replace(' ', '')
        # content_seg = jieba.cut(content)
        # seg_string = " ".join(content_seg)
        # word_list = seg_string.split(' ')
        content = content.strip()
        # content = content.replace(' ', '')
        # 分词
        # content_seg = jieba.cut(content)
        doc = spacy_model(content)
        # items = [([tok.text for tok in sent], sent.text) for sent in doc.sents if len(sent) >= 3]
        word_list = [token.orth_ for token in doc]
        seq = []
        for word in word_list:
            if word in word_index.keys():
                word_seq = word_index[word]
            else:
                word_seq = 0
            seq.append(word_seq)
        seq_list.append(seq)
    seq_predict = sequence.pad_sequences(seq_list, maxlen=1000, padding='post', truncating='post')
    prediction = model.predict(seq_predict)
    result = numpy.argmax(prediction, axis=1)
    id_list = result.tolist()
    result_data = []
    i = 0
    for _id in id_list:
        json_dic = {}
        json_dic['content'] = content_list[i]
        json_dic['class'] = label_list[_id]
        i += 1
        result_data.append(json_dic)
    # json_data = json.dumps(result_data, ensure_ascii=False)
    # return json_data
    return result_data


if __name__ == '__main__':
    spacy_model = init_spacy_model()
    
    # train
    # result = customize_train(r"./data_cp/1130.txt")
    # # val_acc: 0.9610
    # print(result)
    # {"loss": 0.11845778147061302, "precision": 0.8498151831550511, "recall": 0.21967963475384483,
    # "f1": 0.34765827028374924}
    
    # 50 epoch text_cnn
    # result = customize_train(r"./data_cp/1130.txt")
    # val_loss: 0.0717 - val_my_precision: 0.8105 - val_my_recall: 0.7605 - val_my_f1: 0.7846
    
    # lstm
    # result = customize_train(r"./data_cp/1130_lstm.txt")
    # print(result)
    # 5240/5240 [==============================] - 186s 35ms/step - loss: 0.0021 - my_precision: 0.9727 - my_recall: 0.9574 - my_f1: 0.9650 -
    # val_loss: 0.0177 - val_my_precision: 0.8102 - val_my_recall: 0.7292 - val_my_f1: 0.7675
    # 评估。。。
    # 1311/1311 [==============================] - 15s 11ms/step
    # {"loss": 0.018224597927737018, "precision": 0.7952643446027551, "recall": 0.7162471425887922, "f1": 0.753522495246861}
    
    
    
    # test
    # content_list = []
    # with open('tt.txt', 'r', encoding='utf-8') as fr:
    #     content_list.append(fr.readline())
    # model_name = 'default_text-cnn'
    # # model = load_model('neural/default_text-cnn.h5')
    # result = nn_prediction(content_list, model_name)
    # # result = nn_prediction(content_list, model, model_name)
    # print(result)
    
    # content_list = [
    #     "President Trump’s plans to host a summit of leaders of the G-7 group of industrialized nations this month was put to rest "
    #     "when German Chancellor Angela Merkel waved him off. Coronavirus concerns made it impossible for her to confirm her attendance, she said. "
    #     "Trump then announced the gathering would be delayed until September, and if the story had ended there, "
    #     "it wouldn’t have made much news. But this was not going to be an ordinary G-7 summit. "
    #     "Instead, the U.S. President also announced plans to rewrite the guest list. "
    #     "“I don’t feel that as a G-7 it properly represents what’s going on in the world,” Trump told reporters. "
    #     "“It’s a very outdated group of countries.” He has a point. "
    #     "Gone are the days when the U.S., the U.K., France, Germany, Italy, "
    #     "Japan and Canada could credibly claim to represent the world’s advanced economies, much less to set an "
    #     "international agenda. The rise of China, in particular, but also the emergence of countries like India, "
    #     "South Korea, Brazil, Russia, Turkey and others have long made the G-7 look like a country-club board meeting. "
    #     "Trump’s solution: expand the group. “We want Australia, we want India, we want South Korea,” Trump told reporters. "
    #     "“That’s a nice group of countries right there.” As host of this year’s summit, the President has the right to "
    #     "send an invite to whomever he wants. But Trump was proposing what he called a “G-10 or G-11,” "
    #     "a permanent expansion of the group. He has no power to do that by himself. So, what’s the thinking behind the "
    #     "choice of those countries? An invitation to Indo-Pacific democracies India, South Korea and Australia is an "
    #     "attempt to confront and isolate China. If the G-7 is an institution designed to promote democracy; freedoms of "
    #     "speech, assembly and religion; and free-market capitalism, then that idea makes good sense. "
    #     "China has emerged over the past 30 years as the world’s most powerful police state, and its international "
    #     "influence now extends into every region of the world. An alliance of democracies designed to promote and defend "
    #     "democratic values and individual freedoms might be a worthy goal. Unfortunately, Trump’s plan won’t work. "
    #     "First, the President also wants to include Russia. You might recall that Russia was invited to join the G-7 in "
    #     "1997 but was ousted in 2014 in response to its invasion of Ukraine. Vladimir Putin has since made Crimea part "
    #     "of Russia, undermining any support in Europe for Russia’s reinclusion in an alliance designed to promote and "
    #     "protect democracy. Trump’s call for Russia’s return was immediately rejected by both Britain and Canada. "
    #     "If the G-7 summit is held in the U.S. in September, and Russia is invited as a nonmember, prepare for the "
    #     "spectacle of a smiling Putin waving to cameras on the eve of a U.S. election. Then there is the problem of "
    #     "Trump as messenger. The President’s response to ongoing civil unrest in the U.S.–including his threat to "
    #     "use “vicious dogs” and the U.S. military against protesters–undermines his case against China. In this way, "
    #     "he’s made China’s latest messaging much easier. Hong Kong’s Beijing-backed chief administrator pushed back hard "
    #     "on U.S. criticism of China’s bid to impose a Beijing-sanctioned political order on the city: “There are riots in "
    #     "the United States, and we see how local governments reacted. And then in Hong Kong, when we had similar riots, "
    #     "we saw what position they adopted.” There are good counterarguments to make to that, but President Trump is the "
    #     "wrong leader to make them."
    # ]
    # sent_list = []
    # for line in content_list:
    #     text_sentences = spacy_model(line)
    #     for sentence in text_sentences.sents:
    #         print(sentence.text)
    #     # 分句
    #     temp = [l.text for l in text_sentences.sents]
    #     sent_list.extend(temp)
    # # text_cnn
    # model = load_model('neural/1130_text-cnn.h5',
    #                    custom_objects={"my_precision": my_precision, "my_recall": my_recall, "my_f1": my_f1})
    # result = nn_prediction(sent_list, model, "text_cnn", "1130")
    #
    # # lstm
    # # model = load_model('neural/1130_1_lstm.h5',
    # #                    custom_objects={"my_precision": my_precision, "my_recall": my_recall, "my_f1": my_f1})
    # # result = nn_prediction(sent_list, model, "lstm", "1130_1")
    #
    # for s in result:
    #     print(s)
