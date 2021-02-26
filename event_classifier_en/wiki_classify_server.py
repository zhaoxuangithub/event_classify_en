import os
import sys
import json
import prediction
from utils import json_combine, dic_combine
from sklearn.externals import joblib
from flask import Flask, request, Response

sys.path.append(os.path.abspath('..'))

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model_dic = {}
model_dic['default_svm'] = joblib.load('default' + os.sep + 'model' + os.sep + 'svm.m')
model_dic['default_bayes'] = joblib.load('default' + os.sep + 'model' + os.sep + 'bayes.m')
cm_path = 'custom' + os.sep + 'model'
if not os.path.exists(cm_path):
    os.makedirs(cm_path)
model_list = os.listdir(cm_path)
for m in model_list:
    model_key = 'custom' + '_' + m[:-2]
    model_dic[model_key] = joblib.load(cm_path + os.sep + m)
print(model_dic)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/wiki_predict', methods=['POST'])
def wiki_predict():
    json_list = request.json['content']
    model = request.json['model']

    content_list = []
    for json_line in json_list:
        content_list.append(dic_combine(json_line))
    seg_list = prediction.predict_segment(content_list)
    text_bunch = prediction.pack_bunch(seg_list)
    mode = 'custom'
    vec_space = prediction.vectorize(text_bunch, mode, model)
    key = mode + '_' + model
    predict_model = model_dic[key]
    predict_result = predict_model.predict(vec_space.tdm)
    result_data = []
    for file_name, expct_cate in zip(vec_space.filenames, predict_result):
        json_dic = {}
        seq = int(file_name)
        json_dic['content'] = content_list[seq]
        json_dic['class'] = expct_cate
        result_data.append(json_dic)
    json_data = json.dumps(result_data, ensure_ascii=False)

    return Response(json_data, mimetype='application/json')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', threaded=False)
    app.run(host='0.0.0.0', port=5464)  # port为端口号，可自行修改
