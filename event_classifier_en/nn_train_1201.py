import jieba
import os
import numpy
import pickle
from keras.utils import np_utils, multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shutil
import spacy
import json

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
K.clear_session()

# tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

# gpu按需分配
# gpu_id = str(os.environ.get('DEVICE_ID'))
# print("********************gpu_id={}****************".format(gpu_id))
# None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用0号显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用1号显卡
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


class AttentionLayer(Layer):
    """
    自定义Attention
    """
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
        
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class Self_Attention_CNN(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention_CNN, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        hidden_size = input_shape[2]
        if self.output_dim is None:
            self.output_dim = hidden_size
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        
        super(Self_Attention_CNN, self).build(input_shape)  # 一定要在最后调用它
    
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        
        print("WQ.shape", WQ.shape)
        
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
        
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        
        QK = QK / (64 ** 0.5)
        
        QK = K.softmax(QK)
        
        print("QK.shape", QK.shape)
        
        V = K.batch_dot(QK, WV)
        
        return V
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class ParallelModelCheckpoint(ModelCheckpoint):
    """
    自定义一个ModelCheckpoint
    """
    def __init__(self, my_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = my_model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)
    
    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)
    
    
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
    maxlen = 150
    count = 0
    len_list = []
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
            len_list.append(len(content_seg))
            if len(content_seg) > maxlen:
                count += 1
            # print(content, content_seg)
            _savefile(cls_seg_path + single_file, " ".join(content_seg))  # 将处理后的文件保存到分词后语料目录
    print("----------------sentences.len--------------", len(len_list))
    array = numpy.asarray(len_list)
    mean = numpy.mean(array)
    std = numpy.std(array)
    temp_count = int(mean + std) + 1
    print('分词完成！保存结束！mean, std, temp_count == {}, {}, {}'.format(mean, std, temp_count))


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
    # texts_index = sequence.pad_sequences(seq, maxlen=1000, padding='post', truncating='post')
    # # maxlen = 128
    # texts_index = sequence.pad_sequences(seq, maxlen=128, padding='post', truncating='post')
    maxlen = 59
    texts_index = sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    # x_train, x_test, y_train, y_test = train_test_split(texts_index, labels_vec, test_size=0.2)  # 验证集比例
    x_train, x_test, y_train, y_test = train_test_split(texts_index, labels_vec, test_size=val_scale)  # 验证集比例
    if model == 'cnn':
        result = train_cnn(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension)
    elif model == 'lstm':
        result = train_lstm(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension, maxlen=maxlen)
    elif model == 'text_cnn':
        result = text_cnn(model_name, word_num + 1, vec_matrix, x_train, x_test, y_train, y_test, label_num, word_dimension, maxlen=maxlen)
    return result


def create_lstm_attention_model(max_len, vocab_size, embedding_size, hidden_size, attention_size, class_nums):
    """
    attention bi-lstm model create
    :param max_len: 句子最大长度
    :param vocab_size: 字典长度
    :param embedding_size: 词向量长度
    :param hidden_size: lstm units
    :param attention_size: atten size
    :param class_nums: class size
    :return:
    """
    # 输入层
    inputs = Input(shape=(max_len,), dtype='int32')
    # Embedding层
    embedded_sequences = Embedding(vocab_size, embedding_size)(inputs)
    # BiLSTM层
    bi_lstm = Bidirectional(LSTM(hidden_size, dropout=0.2, return_sequences=True))(embedded_sequences)
    # Attention层
    att_layer = AttentionLayer(attention_size=attention_size)(bi_lstm)
    # 输出层
    outputs = Dense(class_nums, activation='softmax')(att_layer)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()  # 输出模型结构和参数数量

    # embedded_sequences = embedding_layer(input)
    # l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    # l_att = AttLayer()(l_lstm)
    # preds = Dense(2, activation='softmax')(l_att)
    # model = Model(sequence_input, preds)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['acc'])
    #
    # print("model fitting - attention GRU network")
    # model.summary()
    # model.fit(x_train, y_train, validation_data=(x_val, y_val),
    #           nb_epoch=10, batch_size=50)
    return model


def train_lstm(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension, maxlen):
    # model = Sequential()
    # model.add(Embedding(input_dim=dim,
    #                     output_dim=word_dimension,
    #                     mask_zero=True,
    #                     weights=[matrix],
    #                     input_length=1000,
    #                     trainable=False))
    # model.add(Bidirectional(LSTM(units=256), merge_mode='sum'))
    # # model.add(Dropout(0.3))
    # model.add(Dropout(0.2))
    # # add attention
    # model.add(AttentionLayer(50))
    # # old
    # # model.add(Dense(label_num, activation='relu'))
    # model.add(Dense(label_num, activation='softmax'))
    
    # 调用attention构建模型
    # model = create_classify_model(1000, dim, word_dimension, 256, 50, label_num)
    # 最大长度定为128,64
    att_size = 50
    unit_size = 320
    model = create_lstm_attention_model(maxlen, dim, word_dimension, unit_size, att_size, label_num)
    print('编译模型。。。')
    # old
    # model.compile(loss='mse',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # 1130
    # model.compile(loss='mse',
    #               optimizer='adam',
    #               metrics=[my_precision, my_recall, my_f1])
    # 1201
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[my_precision, my_recall, my_f1])
    # 1201
    # checkpointer = ModelCheckpoint(os.path.join("./neural", model_name + '_lstm_{epoch:02d}.h5'),
    checkpointer = ModelCheckpoint(os.path.join("./neural", model_name + '_attention_{}_{}_unit_size_{}'.format(att_size, maxlen, unit_size) +
                                                'lstm_{epoch:02d}.h5'),
                                   monitor="val_my_f1", mode="max", save_best_only=True,
                                   save_weights_only=False, period=1)
    print('开始训练。。。')
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=50,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        callbacks=[checkpointer]
                        )
    # model_file = 'neural' + os.sep + model_name + '_lstm.h5'
    # model.save(model_file)
    print('评估。。。')
    # old
    # loss, acc = model.evaluate(x_test, y_test, batch_size=120)
    # 1130
    # loss, precision, recall, f1_s = model.evaluate(x_test, y_test, batch_size=128)
    # 1201
    loss, precision, recall, f1_s = model.evaluate(x_test, y_test, batch_size=64)
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
    print(result)
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


def create_text_cnn_attention_model(max_len, vocab_size, embedding_size, matrix, attention_size, class_nums):
    """
    attention text_cnn model create
    :param max_len: 句子最大长度
    :param vocab_size: 字典长度
    :param embedding_size: 词向量长度
    :param matrix: 索引-向量矩阵
    :param attention_size: output dim
    :param class_nums: class size
    :return:
    """
    # 输入层
    inputs = Input(shape=(max_len,), dtype='int32')
    # Embedding层
    # embedded_sequences = Embedding(vocab_size, embedding_size)(inputs)
    embedding_layer = Embedding(vocab_size,
                                embedding_size,
                                weights=[matrix],
                                input_length=max_len,
                                trainable=False)
    embedded_sequences = embedding_layer(inputs)
    # Attention层
    embedded_sequences = Self_Attention_CNN(attention_size)(embedded_sequences)
    convs = []
    # filter_sizes = [2, 3, 4, 5]
    # TODO 2020-12-02
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        # filters=100
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(embedded_sequences)
        # TODO 1201 attention
        # l_pool = MaxPooling1D(1000 - fsz + 1)(l_conv)
        # max_len --> 128,64
        l_pool = MaxPooling1D(max_len - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merged = concatenate(convs, axis=1)
    # out = Dropout(0.5)(merged)
    # TODO 1201
    out = Dropout(0.3)(merged)
    output = Dense(32, activation='relu')(out)
    # outputs = Dense(units=class_nums, activation='sigmoid')(output)
    # TODO 1201
    outputs = Dense(units=class_nums, activation='softmax')(output)
    model = Model(inputs, outputs)
    model.summary()
    return model


def text_cnn(model_name, dim, matrix, x_train, x_test, y_train, y_test, label_num, word_dimension, maxlen):
    # 1201 old no attention
    # # sequence_input = Input(shape=(1000,), dtype='int32')
    # # max_len --> 128
    # # sequence_input = Input(shape=(128,), dtype='int32')
    # # maxlen = 64
    # sequence_input = Input(shape=(maxlen,), dtype='int32')
    # # embedding_layer = Embedding(dim,
    # #                             word_dimension,
    # #                             weights=[matrix],
    # #                             input_length=1000,
    # #                             trainable=False)
    # # max_len --> 128,64
    # embedding_layer = Embedding(dim,
    #                             word_dimension,
    #                             weights=[matrix],
    #                             input_length=maxlen,
    #                             trainable=False)
    # embedded_sequences = embedding_layer(sequence_input)
    # convs = []
    # filter_sizes = [2, 3, 4, 5]
    # for fsz in filter_sizes:
    #     l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(embedded_sequences)
    #     # l_pool = MaxPooling1D(1000 - fsz + 1)(l_conv)
    #     # max_len --> 128,64
    #     l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
    #     l_pool = Flatten()(l_pool)
    #     convs.append(l_pool)
    # merged = concatenate(convs, axis=1)
    # # out = Dropout(0.5)(merged)
    # # TODO 1201
    # out = Dropout(0.3)(merged)
    # output = Dense(32, activation='relu')(out)
    # # output = Dense(units=label_num, activation='sigmoid')(output)
    # # TODO 1201
    # output = Dense(units=label_num, activation='softmax')(output)
    # model = Model(sequence_input, output)
    
    # TODO 1201 use attention
    att_size = 128
    batch_size = 160
    model = create_text_cnn_attention_model(maxlen, dim, word_dimension, matrix, att_size, label_num)
    
    # old
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # custom
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[my_precision, my_recall, my_f1])

    # 多GPU并行模型
    # sig_model = multi_gpu_model(model, gpus=2)

    # _final_attention_{}_{}_batch_size_{}_'.format(att_size, maxlen, batch_size)
    checkpointer = ModelCheckpoint(os.path.join("./neural", model_name + '_weights_' + 'text_cnn_{epoch:02d}.h5'),
                                   monitor="val_my_f1", mode="max", save_best_only=True,
                                   save_weights_only=True, period=1)
    # use custom checkpoint
    # checkpointer = ParallelModelCheckpoint(model, os.path.join("./neural", model_name + '_final_model_' + 'text_cnn_{epoch:02d}.h5'),
    #                                monitor="val_my_f1", verbose=0, save_best_only=True,
    #                                save_weights_only=False, mode="max", period=1)
    
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
                        nb_epoch=50, batch_size=batch_size, verbose=1,
                        callbacks=[checkpointer]
                        )
    # model_file = 'neural' + os.sep + model_name + '_text-cnn.h5'
    # model.save(model_file)
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
    print(result)
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


def text_cnn_predict(dim, matrix, label_num, word_dimension, maxlen, out_dim):
    """
    构建text_cnn模型
    :param dim: word_num + 1
    :param matrix: 矩阵
    :param label_num: 输出类别数
    :param word_dimension: 词向量维度
    :param maxlen: 句子最大长度
    :return:
    """
    # TODO 1201 use attention
    model = create_text_cnn_attention_model(maxlen, dim, word_dimension, matrix, out_dim, label_num)
    # custom
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[my_precision, my_recall, my_f1])
    return model


def create_model_load_weights(m_type, model_name, weights_path):
    """
    创建模型并加载已经训练后的weights参数返回用于预测的模型
    :param m_type:
    :param model_name:
    :param weights_path:
    :return:
    """
    word_index = pickle.load(open('dictionary' + os.sep + model_name + '_' + m_type + '_' + 'wid.pkl', 'rb'))
    label_path = 'labels' + os.sep + model_name + '_' + m_type + '_labels.txt'
    with open(label_path, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        line = line.rstrip(' ')
        label_list = line.split(' ')
    # 整个字典的词数
    word_num = len(word_index)
    # 词向量字典
    word_dic = {}
    # 词向量维度
    word_dimension = 300
    # 最大句子长度
    maxlen = 59
    # attention output_dim
    output_dim = 128
    # # 索引-向量矩阵
    vec_matrix = numpy.zeros((word_num + 1, word_dimension))
    for word, index in word_index.items():
        if word in word_dic.keys():
            vec_matrix[index] = word_dic[word]
    print('矩阵生成完毕')
    lo_model = text_cnn_predict(word_num + 1, vec_matrix, len(label_list), word_dimension, maxlen, output_dim)
    lo_model.load_weights(weights_path)
    return lo_model, word_index, label_list


def nn_prediction(content_list, spacy_model, model, word_index, label_list):
    """
    text_cnn 预测
    :param content_list:
    :param model: 模型
    :param word_index: 词向量字典
    :param content_list: 类别列表
    :return:
    """
    seq_list = []
    for content in content_list:
        content = content.replace('\r\n', '')
        content = content.strip()
        # 分词
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
    # 调整max_len = 128
    maxlen = 59
    seq_predict = sequence.pad_sequences(seq_list, maxlen=maxlen, padding='post', truncating='post')
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
    # Epoch 33/50  128 max_len
    # 5240/5240 [==============================] - 1s 115us/step - loss: 0.0095 - my_precision: 0.9668 - my_recall: 0.9616 - my_f1: 0.9642
    # - val_loss: 0.0780 - val_my_precision: 0.8164 - val_my_recall: 0.7651 - val_my_f1: 0.7898
    # Epoch 30/50  64 max_len
    # 5240/5240 [==============================] - 0s 84us/step - loss: 0.0096 - my_precision: 0.9680 - my_recall: 0.9628 - my_f1: 0.9654
    # - val_loss: 0.0710 - val_my_precision: 0.8296 - val_my_recall: 0.7712 - val_my_f1: 0.7992
    # Epoch 33/50 64 max_len att_size = 50
    # 5240/5240 [==============================] - 0s 93us/step - loss: 0.0178 - my_precision: 0.9427 - my_recall: 0.9122 - my_f1: 0.9271
    # - val_loss: 0.0627 - val_my_precision: 0.8302 - val_my_recall: 0.7956 - val_my_f1: 0.8124
    # Epoch 35/50 max_len = 64  att_size = 128
    # 5240/5240 [==============================] - 0s 95us/step - loss: 0.0110 - my_precision: 0.9604 - my_recall: 0.9481 - my_f1: 0.9542
    # - val_loss: 0.0764 - val_my_precision: 0.8293 - val_my_recall: 0.8032 - val_my_f1: 0.8160
    # Epoch 31/50 max_len = 59 att_size = 50
    # 5240/5240 [==============================] - 0s 93us/step - loss: 0.0178 - my_precision: 0.9426 - my_recall: 0.9162 - my_f1: 0.9292
    # - val_loss: 0.0660 - val_my_precision: 0.8323 - val_my_recall: 0.7986 - val_my_f1: 0.8151
    # Epoch 33/50 max_len = 59 att_size = 60
    # 5240/5240 [==============================] - 0s 81us/step - loss: 0.0150 - my_precision: 0.9488 - my_recall: 0.9277 - my_f1: 0.9381
    # - val_loss: 0.0686 - val_my_precision: 0.8368 - val_my_recall: 0.7986 - val_my_f1: 0.8172
    # TODO Epoch 22/50 max_len = 59 att_size = 128  best 1201
    # 5240/5240 [==============================] - 0s 92us/step - loss: 0.0181 - my_precision: 0.9416 - my_recall: 0.9177 - my_f1: 0.9294
    # - val_loss: 0.0594 - val_my_precision: 0.8396 - val_my_recall: 0.8124 - val_my_f1: 0.8256
    # Epoch 19/50 max_len = 59 att_size = 150
    # 5240/5240 [==============================] - 1s 99us/step - loss: 0.0209 - my_precision: 0.9338 - my_recall: 0.9006 - my_f1: 0.9168
    # - val_loss: 0.0586 - val_my_precision: 0.8468 - val_my_recall: 0.7979 - val_my_f1: 0.8215
    # Epoch 48/50 max_len = 60 att_size = 128 持平
    # 5240/5240 [==============================] - 1s 96us/step - loss: 0.0091 - my_precision: 0.9650 - my_recall: 0.9580 - my_f1: 0.9615
    # - val_loss: 0.0789 - val_my_precision: 0.8394 - val_my_recall: 0.8169 - val_my_f1: 0.8279
    # Epoch 17/50 max_len = 62 att_size = 128 下降
    # 5240/5240 [==============================] - 0s 93us/step - loss: 0.0235 - my_precision: 0.9284 - my_recall: 0.8893 - my_f1: 0.9083
    # - val_loss: 0.0638 - val_my_precision: 0.8356 - val_my_recall: 0.7918 - val_my_f1: 0.8130
    # Epoch 17/50 max_len = 60 att_size = 150 下降
    # 5240/5240 [==============================] - 1s 101us/step - loss: 0.0207 - my_precision: 0.9367 - my_recall: 0.9002 - my_f1: 0.9180
    # - val_loss: 0.0565 - val_my_precision: 0.8355 - val_my_recall: 0.7941 - val_my_f1: 0.8142
    # Epoch 23/50 max_len = 60 att_size = 138 持平
    # 5240/5240 [==============================] - 1s 99us/step - loss: 0.0158 - my_precision: 0.9483 - my_recall: 0.9239 - my_f1: 0.9359
    # - val_loss: 0.0580 - val_my_precision: 0.8436 - val_my_recall: 0.8162 - val_my_f1: 0.8296
    # TODO Epoch 24/50 max_len = 59 att_size = 128  batch_size = 160 best 1202 1130_attention_128_59_batch_size_160_text_cnn_24.h5
    # 5240/5240 [==============================] - 0s 79us/step - loss: 0.0184 - my_precision: 0.9413 - my_recall: 0.9084 - my_f1: 0.9245
    # - val_loss: 0.0583 - val_my_precision: 0.8499 - val_my_recall: 0.8162 - val_my_f1: 0.8327
    # TODO Epoch 12/50 max_len = 59 att_size = 128  batch_size = 160 best 1202 1130_final_attention_128_59_batch_size_160_text_cnn_12.h5
    # 5240/5240 [==============================] - 0s 74us/step - loss: 0.0339 - my_precision: 0.9059 - my_recall: 0.8345 - my_f1: 0.8687
    # - val_loss: 0.0509 - val_my_precision: 0.8645 - val_my_recall: 0.8002 - val_my_f1: 0.8311
    # Epoch 18/50 max_len = 59 att_size = 128  batch_size = 144 持平 1202
    # 5240/5240 [==============================] - 0s 87us/step - loss: 0.0207 - my_precision: 0.9352 - my_recall: 0.9036 - my_f1: 0.9191
    # - val_loss: 0.0571 - val_my_precision: 0.8545 - val_my_recall: 0.8116 - val_my_f1: 0.8324
    # Epoch 18/50 drop_out = 0.5 max_len = 59 att_size = 128  batch_size = 160 持平
    # 5240/5240 [==============================] - 0s 81us/step - loss: 0.0265 - my_precision: 0.9218 - my_recall: 0.8685 - my_f1: 0.8943
    # - val_loss: 0.0550 - val_my_precision: 0.8574 - val_my_recall: 0.8063 - val_my_f1: 0.8310
    # Epoch 19/50 2 4 6 8 持平 max_len = 59 att_size = 128  batch_size = 160
    # 5240/5240 [==============================] - 0s 82us/step - loss: 0.0206 - my_precision: 0.9356 - my_recall: 0.8996 - my_f1: 0.9172
    # - val_loss: 0.0579 - val_my_precision: 0.8461 - val_my_recall: 0.8146 - val_my_f1: 0.8300
    
    # 测试1203
    # TODO Epoch 40/50 1203 max_len = 59 att_size = 128  batch_size = 160 best
    # 5240/5240 [==============================] - 0s 80us/step - loss: 0.0110 - my_precision: 0.9595 - my_recall: 0.9479 - my_f1: 0.9536
    # - val_loss: 0.0627 - val_my_precision: 0.8514 - val_my_recall: 0.8299 - val_my_f1: 0.8404
    
    
    
    
    
    
    
    
    
    # lstm
    # result = customize_train(r"./data_cp/1130_lstm.txt")
    # print(result)
    # 5240/5240 [==============================] - 186s 35ms/step - loss: 0.0021 - my_precision: 0.9727 - my_recall: 0.9574 - my_f1: 0.9650 -
    # val_loss: 0.0177 - val_my_precision: 0.8102 - val_my_recall: 0.7292 - val_my_f1: 0.7675
    # 评估。。。
    # 1311/1311 [==============================] - 15s 11ms/step
    # {"loss": 0.018224597927737018, "precision": 0.7952643446027551, "recall": 0.7162471425887922, "f1": 0.753522495246861}
    
    # bi-lstm + attention
    # result = customize_train(r"./data_cp/1201.txt")
    # print(result)
    
    # bi-lstm + attention + callbacks save best f1
    # result = customize_train(r"./data_cp/1201.txt")
    # Epoch 38/50 max_len = 128 att_size = 50 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 19s 4ms/step - loss: 0.0019 - my_precision: 0.9747 - my_recall: 0.9611 - my_f1: 0.9678
    # - val_loss: 0.0169 - val_my_precision: 0.8327 - val_my_recall: 0.7132 - val_my_f1: 0.7682
    # Epoch 28/50 max_len = 64  性能下降 att_size = 50 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 11s 2ms/step - loss: 0.0024 - my_precision: 0.9698 - my_recall: 0.9540 - my_f1: 0.9618
    # - val_loss: 0.0200 - val_my_precision: 0.8027 - val_my_recall: 0.6651 - val_my_f1: 0.7273
    # Epoch 44/50 max_len = 59 att_size = 64  下降 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 10s 2ms/step - loss: 0.0019 - my_precision: 0.9805 - my_recall: 0.9529 - my_f1: 0.9664
    # - val_loss: 0.0220 - val_my_precision: 0.7789 - val_my_recall: 0.6545 - val_my_f1: 0.7111
    # Epoch 26/50 max_len = 128 att_size = 64 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 19s 4ms/step - loss: 0.0021 - my_precision: 0.9728 - my_recall: 0.9632 - my_f1: 0.9680
    # - val_loss: 0.0191 - val_my_precision: 0.8060 - val_my_recall: 0.6972 - val_my_f1: 0.7474
    # TODO Epoch 50/50 max_len = 128 att_size = 50 temp_best unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 20s 4ms/step - loss: 0.0019 - my_precision: 0.9777 - my_recall: 0.9523 - my_f1: 0.9647
    # - val_loss: 0.0162 - val_my_precision: 0.8422 - val_my_recall: 0.7445 - val_my_f1: 0.7901
    # Epoch 38/50 max_len = 100 att_size = 50 下降 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 18s 3ms/step - loss: 0.0017 - my_precision: 0.9805 - my_recall: 0.9616 - my_f1: 0.9710
    # - val_loss: 0.0180 - val_my_precision: 0.8196 - val_my_recall: 0.7101 - val_my_f1: 0.7607
    # Epoch 36/50 max_len = 128 att_size = 39 下降 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 21s 4ms/step - loss: 0.0019 - my_precision: 0.9771 - my_recall: 0.9594 - my_f1: 0.9681
    # - val_loss: 0.0177 - val_my_precision: 0.8263 - val_my_recall: 0.7216 - val_my_f1: 0.7701
    # Epoch 29/50 max_len = 128 att_size = 45 下降 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 22s 4ms/step - loss: 0.0021 - my_precision: 0.9729 - my_recall: 0.9574 - my_f1: 0.9651
    # - val_loss: 0.0170 - val_my_precision: 0.8313 - val_my_recall: 0.7246 - val_my_f1: 0.7741
    # Epoch 30/50 max_len = 128 att_size = 55 下降 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 21s 4ms/step - loss: 0.0020 - my_precision: 0.9759 - my_recall: 0.9584 - my_f1: 0.9671
    # - val_loss: 0.0174 - val_my_precision: 0.8218 - val_my_recall: 0.7208 - val_my_f1: 0.7678
    # Epoch 30/50 max_len = 150 att_size = 50 基本持平 unit_size = 256 batch_size = 128
    # 5240/5240 [==============================] - 25s 5ms/step - loss: 0.0022 - my_precision: 0.9726 - my_recall: 0.9548 - my_f1: 0.9636
    # - val_loss: 0.0162 - val_my_precision: 0.8428 - val_my_recall: 0.7429 - val_my_f1: 0.7897
    # Epoch 31/50 max_len = 128 att_size = 50 下降 unit_size = 128 batch_size = 128
    # 5240/5240 [==============================] - 22s 4ms/step - loss: 0.0019 - my_precision: 0.9754 - my_recall: 0.9595 - my_f1: 0.9673
    # - val_loss: 0.0185 - val_my_precision: 0.8142 - val_my_recall: 0.7063 - val_my_f1: 0.7562
    # Epoch 19/50 max_len = 128 att_size = 50 下降 unit_size = 512 batch_size = 128
    # 5240/5240 [==============================] - 22s 4ms/step - loss: 0.0024 - my_precision: 0.9689 - my_recall: 0.9578 - my_f1: 0.9633
    # - val_loss: 0.0163 - val_my_precision: 0.8369 - val_my_recall: 0.7384 - val_my_f1: 0.7845
    # Epoch 35/50 max_len = 128 att_size = 50 下降 unit_size = 320 batch_size = 128
    # 5240/5240 [==============================] - 21s 4ms/step - loss: 0.0021 - my_precision: 0.9737 - my_recall: 0.9527 - my_f1: 0.9631
    # - val_loss: 0.0167 - val_my_precision: 0.8287 - val_my_recall: 0.7384 - val_my_f1: 0.7808
    
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
    
    content_list = [
        "Brian Morgenstern, the deputy communications director, was wearing a jacket with a White House emblem in his office in the "
        "West Wing. The jacket was zipped all the way up, as if he were on his way out. The room, a few doors away from the Oval Office, "
        "was dark, with the shades drawn."

        "His boss, the president, was in another part of the White House. In that moment, Donald Trump was on speaker phone with Rudy "
        "Giuliani, the head of his legal effort to challenge the election, and a group of state lawmakers who had gathered for a "
        "\"hearing\", as they put it, at a hotel in Gettysburg, Pennsylvania."

        "This election was rigged and we can't let that happen,\" the president said on the phone.\""

        "Morgenstern was monitoring the event on his computer screen, in a distracted manner. "
        "A moment later he swivelled in his chair and spoke to a visitor about college, real estate, baseball, and, almost as an "
        "afterthought, the president's achievements."

        "Trump's effort to contest the election results in Pennsylvania failed on Friday, not long after the so-called hearing, "
        "and even that had a shaky legal foundation. An appeals court judge said there was \"no basis\" for his challenge. "
        "A certification of ballots showed President-elect Joe Biden won the state by more than 80,000 votes."
    ]
    sent_list = []
    for line in content_list:
        text_sentences = spacy_model(line)
        for sentence in text_sentences.sents:
            print(sentence.text)
        # 分句
        temp = [l.text for l in text_sentences.sents]
        sent_list.extend(temp)
    # text_cnn
    # model = load_model('neural/1130_attention_128_59_batch_size_160_text_cnn_24.h5',
    #                    custom_objects={"my_precision": my_precision, "my_recall": my_recall, "my_f1": my_f1})
    # result = nn_prediction(sent_list, model, "text_cnn", "1130")

    # lstm
    # model = load_model('neural/1130_1_lstm.h5',
    #                    custom_objects={"my_precision": my_precision, "my_recall": my_recall, "my_f1": my_f1})
    # result = nn_prediction(sent_list, model, "lstm", "1130_1")

    # lstm + attention
    # model = load_model('neural/1201_lstm_38.h5',
    #                    custom_objects={
    #                        "my_precision": my_precision,
    #                        "my_recall": my_recall,
    #                        "my_f1": my_f1,
    #                        "AttentionLayer": AttentionLayer(50)
    #                    })
    # result = nn_prediction(sent_list, model, "lstm", "1201")

    # text_cnn
    # model = load_model('neural/1130_attention_128_59_batch_size_160_text_cnn_24.h5',
    # model = load_model('neural/1130_attention_128_59_batch_size_160_text_cnn_24.h5',
    #                    custom_objects={
    #                        "my_precision": my_precision,
    #                        "my_recall": my_recall,
    #                        "my_f1": my_f1,
    #                        "Self_Attention_CNN": Self_Attention_CNN(128)
    #                    })
    model, word_index, label_list = create_model_load_weights("text_cnn", "1130", r'./neural/1130_weights_text_cnn_40.h5')
    result = nn_prediction(sent_list, spacy_model, model, word_index, label_list)

    for s in result:
        print(s)
