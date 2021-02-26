import os
import sys
import json
import jieba
from sklearn.externals import joblib
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
import pickle
import numpy
from keras.preprocessing import sequence
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *
from keras import backend as K
import spacy

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
sys.path.append(os.path.abspath('.'))
K.clear_session()
# 解决:ValueError: Tensor Tensor("fc1000/Softmax:0", shape=(?, 1000), dtype=float32) is not an element of this graph。
global graph
graph = tf.get_default_graph()

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


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model_dic = {}
# model_dic['default_svm'] = joblib.load('default' + os.sep + 'model' + os.sep + 'svm.m')
# model_dic['default_bayes'] = joblib.load('default' + os.sep + 'model' + os.sep + 'bayes.m')
# cm_path = 'custom' + os.sep + 'model'
# if not os.path.exists(cm_path):
#     os.makedirs(cm_path)
# model_list = os.listdir(cm_path)
# for m in model_list:
#     model_key = 'custom' + '_' + m.rstrip('.m')
#     model_dic[model_key] = joblib.load(cm_path + os.sep + m)
 
# neural_dic = {}
# neural_path = 'neural' + os.sep
# neural_list = os.listdir(neural_path)
# for n in neural_list:
#     model_key = n.rstrip('.h5')
#     K.backend.clear_session()
#     temp_model = load_model(neural_path + n)
#     temp_model.predict(numpy.zeros((1, 1000)))
#     neural_dic[model_key] = temp_model
#     # neural_dic[model_key] = load_model(neural_path + n)


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


def init_spacy_model():
    """
    初始化spacy model
    """
    # nlp_name = "en_core_sci_sm"
    # nlp = spacy.load(nlp_name)
    nlp = spacy.load('en_core_web_sm')
    # en_core_sci_sm下有模型进行加载
    return nlp


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


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
    # 解决:ValueError: Tensor Tensor("fc1000/Softmax:0", shape=(?, 1000), dtype=float32) is not an element of this graph。
    with graph.as_default():
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


# @app.route('/train', methods=['POST'])
# def train():
# 	if request.method == 'POST':
# 		file = request.files['file']
# 		if file and allowed_file(file.filename):
# 			filename = secure_filename(file.filename)
# 			if not os.path.exists(UPLOAD_FOLDER):
# 				os.makedirs(UPLOAD_FOLDER)
# 			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# 			file.save(filepath)
# 			print('文件已接收，开始训练自定义模型')
# 		data = customize_model.custom(filepath)
# 		return Response(data, mimetype='application/json')


# @app.route('/predict', methods=['POST'])
# def predict_service():
# 	content_list = request.json['content']
# 	model = request.json['model']
# 	default = request.json['default']
#
# 	result_data = prediction.fast_prediction(content_list, default, model)
# 	json_data = json.dumps(result_data, ensure_ascii=False)
# 	return Response(json_data, mimetype='application/json')


# @app.route('/fast_predict', methods=['POST'])
# def fast_predict_service():
# 	content_list = request.json['content']
# 	model = request.json['model']
# 	default = request.json['default']
#
# 	seg_list = prediction.predict_segment(content_list)
# 	text_bunch = prediction.pack_bunch(seg_list)
#
# 	if default:
# 		mode = 'default'
# 		vec_space = prediction.vectorize(text_bunch, mode)
# 	else:
# 		mode = 'custom'
# 		vec_space = prediction.vectorize(text_bunch, mode, model)
#
# 	key = mode + '_' + model
# 	predict_model = model_dic[key]
# 	predict_result = predict_model.predict(vec_space.tdm)
# 	result_data = []
# 	for file_name, expct_cate in zip(vec_space.filenames, predict_result):
# 		json_dic = {}
# 		seq = int(file_name)
# 		json_dic['content'] = content_list[seq]
# 		json_dic['class'] = expct_cate
# 		result_data.append(json_dic)
# 	json_data = json.dumps(result_data, ensure_ascii=False)
#
# 	return Response(json_data, mimetype='application/json')

spacy_model = None
model = None
word_index = None
label_list = None


@app.route('/event_dist_en', methods=['POST'])
def fast_prediction():
    """
    英文事件识别
    :return:
    """
    global spacy_model, model, word_index, label_list
    content_list = request.json['content']
    sent_list = []
    for line in content_list:
        text_sentences = spacy_model(line)
        # for sentence in text_sentences.sents:
        # 	print(sentence.text)
        # 分句
        temp = [l.text for l in text_sentences.sents]
        sent_list.extend(temp)
    if sent_list:
        predict_result = nn_prediction(sent_list, spacy_model, model, word_index, label_list)
    else:
        predict_result = []
    result_data = []
    for js in predict_result:
        if js["class"] != "无":
            result_data.append(js)
    json_data = json.dumps(result_data, ensure_ascii=False)

    return Response(json_data, mimetype='application/json')


# @app.route('/get_model', methods=['POST'])
# def get_model():
# 	model_list = customize_model.get_custom_model()
# 	json_data = json.dumps(model_list)
# 	return Response(json_data, mimetype='application/json')


# @app.route('/train_nn', methods=['POST'])
# def create_nn_model():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             if not os.path.exists(UPLOAD_FOLDER):
#                 os.makedirs(UPLOAD_FOLDER)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             print('文件已接收，开始训练神经网络模型')
#         data = nn_train.customize_train(filepath)
#         return Response(data, mimetype='application/json')

# 神经网络预测，因为keras版本不对，目前没法用
# @app.route('/nn_predict', methods=['POST'])
# def nn_predict():
#     content_list = request.json['content']
#     model_name = request.json['model_name']
#     model = neural_dic[model_name]
#     json_data = nn_train.nn_prediction(content_list, model, model_name)
#     return Response(json_data, mimetype='application/json')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', threaded=False)
    # 初始化spacy
    spacy_model = init_spacy_model()
    model, word_index, label_list = create_model_load_weights("text_cnn", "1130", r'./neural/1130_weights_text_cnn_40.h5')
    app.run(host='0.0.0.0', port=4447)  # port为端口号，可自行修改

