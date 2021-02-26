"""
测试读取excel
"""
import xlrd
import re
import xlwt
import json

pathexcel = r'D:/Work/TNT系统/算法测试素材/TNT-测试文本-20201022-1.xlsx'
pathtartxt = r'D:\Work\事件抽取\测试数据\zh\test_20200806.txt'
pathtartxt2 = r'D:\Work\事件抽取\测试数据\zh\test_split_20200806.txt'


def excel(sourcepath):
	"""
	读取excel文件
	:params sourcepath 测试excel文件路径
	:return: 返回需要列的内容列表
	"""
	# 打开Excel文件
	wb = xlrd.open_workbook(sourcepath)
	# 通过excel表格名称(工作表1)获取工作表
	# sheet = wb.sheet_by_name('工作表1')
	sheet = wb.sheet_by_name('Sheet1')
	# 创建空list
	datas = []
	lang_indies = []
	# 循环读取表格内容（每次读取一行数据）
	for i, a in enumerate(range(sheet.nrows)):
		# 每行数据赋值给cells
		cells = sheet.row_values(a)
		# print(type(cells))
		# list
		# print(i, cells)
		# 0 ['编号', '发布日期', '标题', '媒体', '正文', '媒体国家', '语种', '事件类别', '网址']
		if i == 0:
			# 过滤第一行标题行
			continue
		
		# 因为表内可能存在多列数据，0代表第一列数据，1代表第二列，以此类推
		index = cells[0]
		data = cells[4]
		lang = cells[6]
		# 把每次循环读取的数据插入到list
		# 过滤长度短的干扰项，和重复项
		if len(data) < 5:
			continue
		else:
			if data not in datas:
				datas.append(data)
				lang_indies.append((lang, index))
	assert len(datas) == len(lang_indies), 'length not equal'
	return datas, lang_indies
	

# def readexc2txt():
# 	"""
# 	读取excel文件，并取每行第一列并写入txt文档中
# 	:return:
# 	"""
# 	# 返回整个函数的值
# 	a = excel()
# 	# print(len(a))
# 	with open(pathtartxt, 'at', encoding='utf-8') as fw:
# 		for line in a:
# 			fw.write(line + '\n')
			
		
def switch_dq_and(txt):
	"""
	将双引号中的句子换成为& 防止拆分句子时出现问题
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
			
			
def readtxtsplit(sourcepath):
	"""
	分割句子并避免双引号被拆分 返回拆分后的句子并去重最后返回
	:return:
	"""
	res = []
	with open(sourcepath, 'rt', encoding='utf-8') as fr:
		for line in fr:
			s = line.strip()
			if s:
				# 替换并提取双引号句子
				txt, dquos = switch_dq_and(s)
				# 对处理后的文本进行分句
				# re 分句2
				temps = cut_sent(txt)
				# 将分句后的结果再将占位符替换回原来的双引号句子便于后续处理
				if dquos:
					sentences = redu_sent(dquos, temps)
				else:
					sentences = temps
				if sentences:
					temp = [sent for sent in sentences if sent not in res]
					if temp:
						res.extend(temp)
	return res


def split_txt(txt: str):
	"""
	长文本分句进行句子拆分并返回拆分后的句子列表
	"""
	print(txt)
	# 替换并提取双引号句子
	txt, dquos = switch_dq_and(txt)
	# 对处理后的文本进行分句
	# re 分句
	temps = cut_sent(txt)
	print(temps)
	temps = [line.strip() for line in temps if line and line.strip()]
	print(temps)
	# 将分句后的结果再将占位符替换回原来的双引号句子便于后续处理
	if dquos:
		sentences = redu_sent(dquos, temps)
	else:
		sentences = temps
	return sentences


def write2txt(lst, targetpath):
	"""
	将列表内容逐行写入文件中
	:param lst: 列表
	:param targetpath: 目标文件路径
	:return:
	"""
	if lst:
		with open(targetpath, 'at', encoding='utf-8') as fw:
			for line in lst:
				fw.write(line + '\n')
				
		
def write2excel(file_path, datas):
	"""
	将数据写入新excel文件
	:param file_path:
	:param datas:
	:return:
	"""
	f = xlwt.Workbook()
	# 创建sheet
	sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
	# 将数据写入第 i 行，第 j 列
	for i, data in enumerate(datas):
		for j in range(len(data)):
			sheet1.write(i, j, data[j])
	# 保存文件
	f.save(file_path)
	
	
def dic2list(path):
	"""
	将json格式txt文件读出。每个json封装成一个列表并添加到外层列表中返回
	:param path:
	:return:
	"""
	res = []
	res.append(['事件类型', '触发词', '主体', '客体', '来源'])
	with open(path, 'rt', encoding='utf-8') as fr:
		for line in fr:
			s = line.strip()
			if s and s.startswith('{') and s.endswith('}'):
				js = json.loads(s)
				temp = [js['事件类型'], js['触发词'], js['主体'], js['客体'], js['sentence']]
				res.append(temp)
	if len(res) > 1:
		return res
	return []


if __name__ == '__main__':
	# # lst = readtxtsplit(pathtartxt)
	# # write2txt(lst, pathtartxt2)
	# path = r'D:\Work\事件抽取\测试数据\zh\result_20200806.txt'
	# # targetpath = r'D:\Work\事件抽取\测试数据\zh\result_20200806.xlsx'
	# targetpath = r'D:\Work\事件抽取\测试数据\zh\result_20200806.xls'
	# res = dic2list(path)
	# # print(res)
	# write2excel(targetpath, res)
	
	datas, lang_indies = excel(pathexcel)
	print(lang_indies)
