import pickle
import os
import re
import shutil
import json
# import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.externals import joblib
import spacy


import prediction

temp_custom_path = r'D:/Pycharm_workspace/web_classifier_custom_data_en/custom'


def init_spacy_model():
	"""
	初始化spacy model
	"""
	nlp_name = "en_core_sci_sm"
	nlp = spacy.load(nlp_name)
	# en_core_sci_sm下有模型进行加载
	return nlp


def _dircheck(path):
	"""
	检查文件路径如果没有则创建
	:param path:
	:return:
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def _readfile(path):
	"""
	读文件
	:param path:
	:return:
	"""
	with open(path, 'r', encoding='utf-8') as fr:
		content = fr.read()
		return content


def _savefile(savepath, content):
	"""
	写入文件
	:param savepath:
	:param content:
	:return:
	"""
	with open(savepath, "w", encoding='utf-8') as fw:
		fw.write(content)


def _readbunchobj(path):
	"""
	加载Bunch对象, 通过pickle.load 通过二进制方式读出
	:param path:
	:return:
	"""
	with open(path, "rb") as file_obj:
		bunch = pickle.load(file_obj)
	return bunch


def _writebunchobj(path, bunchobj):
	"""
	将Bunch对象写入文件中,通过pickle.dump 以二进制对象写入
	:param path:
	:param bunchobj:
	:return:
	"""
	with open(path, "wb") as file_obj:
		pickle.dump(bunchobj, file_obj)


'''
文件第一行提供训练的参数
model：选择的模型，目前有svm和bayes
model_name:模型最后保存的名字***_svm.m
scale：测试集占整个文本集的比例
'''


def analyze_file(path):
	"""
	分析文件并做相关处理
	:param path:
	:return:
	"""
	# 语料文件夹
	cp = temp_custom_path + os.sep + 'corpus'
	# 分段路径
	sp = temp_custom_path + os.sep + 'segment'
	if os.path.exists(cp):
		# 递归删除一个目录以及目录内的所有内容
		shutil.rmtree(cp)
	if os.path.exists(sp):
		shutil.rmtree(sp)
	with open(path, 'r', encoding='utf-8') as fr:
		first = True
		num = 1
		for line in fr:
			if first:
				params = json.loads(line)
				model = params['model']
				model_name = params['model_name']
				scale = float(params['scale'])
				first = False
				continue
			data = json.loads(line)
			try:
				content = data['content']
			except KeyError:
				print(data)
			classification = data['class']
			path = temp_custom_path + os.sep + 'corpus' + os.sep + classification
			_dircheck(path)
			file_path = path + os.sep + str(num) + '.txt'
			num += 1
			fw = open(file_path, 'w', encoding='utf-8')
			fw.write(content)
			fw.close()
	print('文件分类保存完成')
	return model, model_name, scale


def corpus_seg(scale):
	"""
	分词并根据测试集占比，划分训练集和测试集
	:param scale:
	:return:
	"""
	corpus_path = temp_custom_path + os.sep + 'corpus'
	clslist = os.listdir(corpus_path)
	for cls in clslist:
		# 遍历不同类别的文件夹
		# 构建语料文件夹路径
		class_path = corpus_path + os.sep + cls + os.sep
		train_path = temp_custom_path + os.sep + 'segment' + os.sep + 'train' + os.sep
		test_path = temp_custom_path + os.sep + 'segment' + os.sep + 'test' + os.sep
		_dircheck(train_path)
		_dircheck(test_path)
		# 训练集文件路径
		train_seg_path = train_path + cls + os.sep
		# 测试集文件路径
		test_seg_path = test_path + cls + os.sep
		_dircheck(train_seg_path)
		_dircheck(test_seg_path)

		one_cls_list = os.listdir(class_path)
		# 测试集数量
		test_num = int(len(one_cls_list)*scale)
		count = 0
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
			if count < test_num:
				# 将处理后的文件保存到分词后语料目录
				# 空格拼接分词后的结果
				_savefile(test_seg_path + single_file, " ".join(content_seg))
			else:
				_savefile(train_seg_path + single_file, " ".join(content_seg))
			count += 1
	print('分词完成！划分测试集结束！')


# model指使用svm或者bayes， model_name指用户自定义的该模型名字， t_type指的是用于训练还是测试(train/test)
def pack_bunch(model, model_name, t_type):
	# 构建测试，训练集的Bunch对象并分别写入各自对应的文件中
	wordbag_root = 'custom' + os.sep + 'bunch' + os.sep
	wordbag_path = wordbag_root + model_name + '_' + model + '_' + t_type + '.dat'
	# 划分的测试集或者训练集路径
	seg_path = temp_custom_path + os.sep + 'segment' + os.sep + t_type
	_dircheck(wordbag_root)
	_dircheck(seg_path)
	# 类别名称列表
	catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
	# 创建一个Bunch实例
	bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
	# 类别名称列表
	bunch.target_name.extend(catelist)
	# 获取每个目录下所有的文件
	for mydir in catelist:
		class_path = seg_path + os.sep + mydir + os.sep  # 拼出分类子目录的路径
		file_list = os.listdir(class_path)  # 获取class_path下的所有文件
		for file_path in file_list:  # 遍历类别目录下文件
			fullname = class_path + file_path  # 拼出文件名全路径
			bunch.label.append(mydir)
			bunch.filenames.append(fullname)
			bunch.contents.append(_readfile(fullname))  # 读取文件内容
	# 将bunch存储到wordbag_path路径中
	_writebunchobj(wordbag_path, bunch)
	print("构建文本对象结束！文件保存在：" + wordbag_path)


def vector_space(model, model_name, t_type):
	"""
	构建词向量并构建Bunch对象，然后写入文件
	tfidf向量使用sklearn库来实现的
	:param model:
	:param model_name:
	:param t_type:
	:return:
	"""
	bunch_path = 'custom' + os.sep + 'bunch' + os.sep + model_name + '_' + model + '_' + t_type + '.dat'
	vec_path = 'custom' + os.sep + 'tfidf' + os.sep
	_dircheck(vec_path)
	train_vec = vec_path + model_name + '_' + model + '_' + 'train' + '.dat'
	test_vec = vec_path + model_name + '_' + model + '_' + 'test' + '.dat'
	stopword_path = 'stopword' + os.sep + 'stopwords.txt'
	# 中英文停止词列表
	stpwrdlst = open(stopword_path, encoding='utf-8').read().replace(r'\n', ' ').split()
	# 读取训练或者测试集的 文本Bunch对象
	bunch = _readbunchobj(bunch_path)
	# 创建 tfidf向量空间 Bunch对象
	tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames,
					   tdm=[], vocabulary={})

	if t_type == 'test':
		trainbunch = _readbunchobj(train_vec)
		tfidfspace.vocabulary = trainbunch.vocabulary
		vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
									 vocabulary=trainbunch.vocabulary)
		tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
		_writebunchobj(test_vec, tfidfspace)
	else:
		vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
		tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
		tfidfspace.vocabulary = vectorizer.vocabulary_
		_writebunchobj(train_vec, tfidfspace)
	print("tf-idf词向量空间实例创建成功！")


def train(model, model_name):
	"""
	传入模型类别model 和 要保存的模型名称进行对应训练 返回得分.
	:param model:
	:param model_name:
	:return:
	"""
	train_vec = 'custom' + os.sep + 'tfidf' + os.sep + model_name + '_' + model + '_' + 'train' + '.dat'
	test_vec = 'custom' + os.sep + 'tfidf' + os.sep + model_name + '_' + model + '_' + 'test' + '.dat'
	model_path = 'custom' + os.sep + 'model' + os.sep
	# 要保存的模型路径全名
	model_file = model_path + model_name + '_' + model + '.m'
	# 路径检查
	_dircheck(model_path)
	# 读取训练测试tfidf词向量对象
	train_set = _readbunchobj(train_vec)
	test_set = _readbunchobj(test_vec)
	if model == 'svm':
		clf = SVC(kernel='linear')
	elif model == 'bayes':
		clf = MultinomialNB(alpha=0.001)

	clf.fit(train_set.tdm, train_set.label)  # 训练分类器
	joblib.dump(clf, model_file)  # 保存模型

	predicted = clf.predict(test_set.tdm)  # 测试集预测
	score = {}
	score['name'] = model_name + '_' + model
	score['save_path'] = model_file
	score['accuracy'] = '{0:.3f}'.format(metrics.precision_score(test_set.label, predicted, average='weighted'))
	score['recall'] = '{0:0.3f}'.format(metrics.recall_score(test_set.label, predicted, average='weighted'))
	score['f1-score'] = '{0:.3f}'.format(metrics.f1_score(test_set.label, predicted, average='weighted'))
	print('精度:' + score['accuracy'])
	print('召回:' + score['recall'])
	print('f1-score:' + score['f1-score'])
	return score


def custom(path):
	# 解析文件各类别分成一条一文件保存下来
	model, model_name, scale = analyze_file(path)
	# 分词并划分测试训练集
	corpus_seg(scale)
	# 构建Bunch对象并写入文件
	pack_bunch(model, model_name, 'train')
	pack_bunch(model, model_name, 'test')
	# 构建tfidf词向量 Bunch对象并写入文件
	vector_space(model, model_name, 'train')
	vector_space(model, model_name, 'test')
	# 训练
	score = train(model, model_name)
	# 返回结果dict
	json_data = json.dumps(score)
	return json_data


def get_custom_model():
	cm_path = 'custom' + os.sep + 'model'
	model_list = os.listdir(cm_path)
	return model_list


def train_test_0703():
	model, model_name, scale = analyze_file('D:/工作2020/train_0702.txt')
	corpus_seg(scale)
	pack_bunch(model, model_name, 'train')
	pack_bunch(model, model_name, 'test')
	vector_space(model, model_name, 'train')
	vector_space(model, model_name, 'test')
	score = train(model, model_name)
	json_data = json.dumps(score)
	return json_data


def switch_dq_and(txt):
	"""
	将双引号中的句子换成为&& 防止拆分句子时出现问题
	并返回提取后的句子列表和替换后的文本
	:param txt:
	:return:
	"""
	txt = txt.replace(r'“”', "")
	txt = txt.replace(r'""', "")
	sents = []
	while '“' in txt and '”' in txt:
		s = txt.find('“')
		e = txt.find('”')
		if s < e:
			e += 1
			se = txt[s:e]
			# print(se)
			sents.append(se)
			txt = txt.replace(se, '&&', 1)
			# print(txt)
		else:
			break
	return txt, sents


def cut_sent(para):
	"""re 分句"""
	para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
	para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
	para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
	para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
	# 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
	para = para.rstrip()  # 段尾如果有多余的\n就去掉它
	# 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
	# return [line for line in para.split("\n") if line and len(line.strip()) >= 5]
	return para.split("\n")


def redu_sent(dqs, sentences):
	"""
	根据之前提取出的字符串列表和切分后的句子列表根据特殊符号&进行句子还原
	:param dqs: 提取的双引号句子列表
	:param sentences: 句子拆分结果列表
	:return:
	"""
	ressents = []
	index = 0
	for s in sentences:
		temp = s
		while '&&' in temp and index < len(dqs):
			temp = temp.replace('&&', dqs[index], 1)
			index += 1
		ressents.append(temp)
	return ressents


def split_sentences_deal_dq(text):
	"""
	将text文章拆分成句子保留双引号中的句子为整体
	:param text:
	:return:
	"""
	txt, dquos = switch_dq_and(text)
	# 对处理后的文本进行分句
	# re 分句2
	temps = cut_sent(txt)
	# 将分句后的结果再将占位符替换回原来的双引号句子便于后续处理
	if dquos:
		sentences = redu_sent(dquos, temps)
	else:
		sentences = temps
	# 处理长度小于5的
	if sentences:
		sentences = [t.strip() for t in sentences if len(t) >= 5]
	return sentences


if __name__ == '__main__':
	# result = train_test_0703()
	# print(result)
	spacy_model = init_spacy_model()

	# 1. 训练
	# test.py处理数据
	
	# 精度:0.971
	# 召回:0.969
	# f1-score:0.970
	# custom(r'./data_cp/1014.txt')
	# 精度:0.957
	# 召回:0.955
	# f1-score:0.955
	# custom(r'./data_cp/1022.txt')
	# 精度:0.940
	# 召回:0.937
	# f1-score:0.937
	# custom(r'./data_cp/1028_15.txt')
	# 精度: 0.655
	# 召回: 0.622
	# f1 - score: 0.604
	# custom(r'./data_cp/1103_8_en_deal.txt')
	# 精度:0.754
	# 召回:0.718
	# f1-score:0.699
	# custom(r'./data_cp/1104_test.txt')
	# 精度:0.996
	# 召回:0.996
	# f1-score:0.996
	# custom(r'./data_cp/1104_test2.txt')
	# 精度:0.953
	# 召回:0.952
	# f1-score:0.952
	# custom(r'./data_cp/1104_test3.txt')
	# 精度:0.938
	# 召回:0.934
	# f1-score:0.932
	# custom(r'./data_cp/1103_8_en_deal.txt')  # 无 抗议
	# 精度:0.780
	# 召回:0.737
	# f1-score:0.722
	# custom(r'./data_cp/1121.txt')  # 英文8类测试
	# 精度:0.894
	# 召回:0.893
	# f1-score:0.893
	# TODO 2020-11-28
	# custom(r'./data_cp/1128.txt')  # 英文20+1类测试
	# 精度:0.747
	# 召回:0.732
	# f1-score:0.733
	custom(r'./data_cp/1128_1.txt')  # 英文20+1类测试 去重
	# 精度:0.790
	# 召回:0.780
	# f1-score:0.780

	# 2.
	# # content_list = ['约翰·海因里希·冯·杜能（Johann Heinrich von Thünen，1783年6月24日－1850年9月22日），台湾多译为邱念[1]，香港译为范杜能 ，
	# 又有译屠能[3]，梅克伦堡经济学者。他的学说被认为是经济地理学和农业地理学的开创者。他被费尔南·布劳岱尔称为除了马克思之外十九世纪的最伟大的经济学者。']
	# content_list = ['《星光大道》是中央电视台综艺频道推出的一档选秀节目，由朱迅、尼格买提搭档主持。 [1] 节目不同于其他娱乐节目以明星表演为主的局面，
	# 本着“百姓自娱自乐”的宗旨，突出大众参与性、娱乐性，力求为全国各地，各行各业的普通劳动者提供一个放声歌唱，展现自我的舞台。
	# 节目于2004年10月9日起每周六晚22:30在中央电视台综艺频道首播 [2]  ，2013年1月5日起每周六晚20:05在中央电视台综合频道首播。2019年4月24日起，每周五22:38在中央电视台综合频道首播']
	
	# content_list = ['国际金融中心（英语：Financial centre），指以第三级产业经济为主；以金融业服务业为中心的全球城市，'
	#                 '这个全球城市必须拥有跨国公司和国际大银行的总部设立，要有活跃的外汇市场、股票市场、期货交易、证券市场等金融产品市场，并拥有至少一个证券交易所。'
	#                 '此外，还需要完善的法律制度和资本主义环境，并有著健全的交通运输、人才教育等硬件建设与体系制度。\n截至2018年9月12日，全球金融中心指数排名为[1]：\n\n\n\n。'
	#                 '习主席说：“加强共产党领导，一党专政，多党合作。”，提高共产党领导能力。1997年12月9日，台湾环境保护联盟发表了支持台中拜耳案公民投票声明，全力支持拜耳案公投。'
	#                 '当晚，蓬贝和巴育接连发表讲话，宣布泰国进入拉玛十世时代，宫务处发布了拉玛十世的头衔名号为泰语：มหาวชิราลงกรณ บดินทรเทพยวรางกูร'
	#                 '（Mahawachiralongkon Bodinthrathepphayawarangkun）[8]。1970年12月22日，召开华北会议，周恩来主持会议，表面上批判陈伯达'
	#                 '及其在华北地区的追随者，但实际上改组了北京军区的领导班子：撤换了北京军区司令员和第二政委，还有38军也被调离了北京地区。'
	#                 '然而在3月4日举行的第2次众议院投票中，该提议以131票赞成（主要是工人社会党、公民党和加那利联盟党议员）、219票反对的结果被驳回[34]。'
	#                 '光照会的成员把难以控制力量的浩克流放到太空；由于反变种人运动和一系列英雄的错误，美国通过了超人类注册法案，美国队长和钢铁人在争论过程中对立，导致大事件“内战”。'
	#                 '2012年6月7日丹麦国会投票通过同性婚姻法案，该法案于6月15日生效。1964年法国承认中华人民共和国并与之建交，中华民国与法国断交，此后法国开始派遣驻中华人民共和国大使。'
	#                 '1850年，君士坦丁堡牧首正式承认希腊东正教会的独立自主地位。今年NBA联赛的总冠军是湖人队。']
	# # model = '1014_svm'
	# # model = '1022_svm'
	# model = '1028_15_svm'
	# default = False
	# res = []
	# for content in content_list:
	# 	temps = cut_sent(content)
	# 	if temps:
	# 		res.extend(temps)
	# if res:
	# 	result_data = prediction.fast_prediction(res, default, model)
	# else:
	# 	result_data = []
	# print(result_data)
	
	# 3.读取文件获取文章并进行分句和判断
	# s_path = r'D:/Work/TNT系统/算法测试素材/测试素材-2-zh.txt'
	# t_path = r'D:/Work/TNT系统/算法测试素材'
	# classify_filename = "测试素材-2-zh-classify.txt"
	# result = []
	# with open(s_path, 'r', encoding='utf-8') as fr:
	# 	for line in fr:
	# 		if line and line.strip():
	# 			js = json.loads(line.strip())
	# 			num = js["index"]
	# 			content = js["content"]
	# 			model = '1022_svm'
	# 			default = False
	# 			res = split_sentences_deal_dq(content)
	# 			if res:
	# 				result_data = prediction.fast_prediction(res, default, model)
	# 			else:
	# 				result_data = []
	# 			print(result_data)
	# 			if result_data:
	# 				for d in result_data:
	# 					# print(d)
	# 					d["index"] = num
	# 					js_str = json.dumps(d, ensure_ascii=False) + "\n"
	# 					# print(js_str)
	# 					if js_str not in result:
	# 						result.append(js_str)
	# if result:
	# 	# print(result)
	# 	with open(os.path.join(t_path, classify_filename), 'w', encoding='utf-8') as fw:
	# 		for line in result:
	# 			# print(line)
	# 			fw.write(line)
