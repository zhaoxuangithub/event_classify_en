"""
post请求测试
"""
import json
import urllib.request
import urllib.parse
import logging
import os

from pri_test_read_excel import excel, write2excel, split_txt


def list_files(p_path):
	"""
	根据path获取路径下所有需要的文件名
	:param p_path:
	:return: list of names
	"""
	if os.path.exists(p_path):
		files = os.listdir(p_path)
	else:
		return FileNotFoundError('%s IS NOT FOUND' % p_path)
	files = [s for s in files if s.startswith('wiki')]
	return files


def post_inform(_id, url, content_text, tpath):
	url = url
	data = json.dumps(content_text)
	data = bytes(data, 'utf8')
	# print(data)
	headers = {"Content-Type": 'application/json'}
	req = urllib.request.Request(url=url, headers=headers, data=data)
	try:
		resp = urllib.request.urlopen(req).read()
		# print(resp.decode('utf-8'))
		resp_dic = json.loads(resp.decode('utf-8'))
		content_array = resp_dic["content"]
		with open(tpath, 'a', encoding='utf-8') as fw:
			for ary in content_array:
				print(ary)
				ary["id"] = _id
				fw.write(json.dumps(ary, ensure_ascii=False) + "\n")
	except Exception as e:
		logging.error(e)
		print(e)
		print(content_text)
		
		
def post_event_extraction(num, url, content_text):
	data = json.dumps(content_text).encode(encoding='utf-8')
	# data = bytes(data, 'utf8')
	# print(data)
	headers = {"Content-Type": 'application/json'}
	req = urllib.request.Request(url=url, headers=headers, data=data)
	try:
		resp = urllib.request.urlopen(req).read()
		# print(resp.decode('utf-8'))
		resp_dic = json.loads(resp.decode('utf-8'))
		content_array = resp_dic["content"]
		print(content_array)
		for ary in content_array:
			ary["number"] = num
		return content_array
	except Exception as e:
		logging.error(e)
		print(e)
		print(content_text)
		return list()


def get_title(sub, t, ob, lan):
	"""
	根据要素形成 标题
	sub, trigger, ob
	"""
	str_sub = ''
	if sub:
		if lan == 'zh':
			str_sub = '、'.join(sub)
		elif lan == 'en':
			str_sub = ','.join(sub)
	str_ob = ''
	if ob:
		if lan == 'zh':
			str_ob = '、'.join(ob)
		elif lan == 'en':
			str_ob = ','.join(ob)
	if str_sub or str_ob:
		if lan == 'zh':
			title = str_sub + " " + t + str_ob
		elif lan == 'en':
			title = ''
			if str_sub:
				title += str_sub + " " + t
			if str_ob:
				if title:
					title += " " + str_ob
				else:
					title += t + " " + str_ob
	else:
		title = ''
	return title


def format_list(ary, lan):
	"""
	格式化ary
	lan: zh 中文简体/ en 英语
	"""
	# 编号、事件类型、触发词、事件标题、相关实体、线索（句）
	# {'text': '新华社北京10月21日电  国务院总理李克强10月21日主持召开国务院常务会议，要求进一步抓好财政资金直达机制落实，
	# 更好发挥积极财政政策效能；决定全面推行证明事项和涉企经营许可事项告知承诺制，以改革更大便利企业和群众办事创业。',
	# 'number': 1,
	# 'event_list': [{'trigger': '主持召开', 'event_type': '会议活动',
	# 'arguments': {'subject': ['李克强'], 'time': ['10月21日'], 'object': ['国务院常务会议']}}]}
	res = []
	if len(ary) > 0:
		res.append(['编号', '事件类型', '触发词', '事件标题', '相关实体', '线索'])
		for d in ary:
			event_list = d['event_list']
			for e in event_list:
				temp_lst = []
				temp_lst.append(d['number'])
				temp_lst.append(e['event_type'])
				temp_lst.append(e['trigger'])
				arg_obj = e['arguments']
				sub = []
				if 'subject' in arg_obj:
					sub = arg_obj['subject']
				t = []
				if 'time' in arg_obj:
					t = arg_obj['time']
				ob = []
				if 'object' in arg_obj:
					ob = arg_obj['object']
				title = get_title(sub, e['trigger'], ob, lan)
				# title = '、'.join(sub) + e['trigger'] + '、'.join(ob)
				temp_lst.append(title)
				entities = []
				entities.extend(t)
				entities.extend(sub)
				entities.extend(ob)
				if lan == 'zh':
					entities = '；'.join(entities)
				elif lan == 'en':
					entities = ';'.join(entities)
				temp_lst.append(entities)
				temp_lst.append(d['text'])
				res.append(temp_lst)
	return res
		
	
def write2file(lst: list, filepath: str):
	"""
	将列表中的内容写入文件中
	"""
	with open(filepath, 'w', encoding='utf-8') as fw:
		for ary in lst:
			print(ary)
			fw.write(json.dumps(ary, ensure_ascii=False) + "\n")


if __name__ == '__main__':
	# pathexcel = r'D:/Work/TNT系统/算法测试素材/TNT-测试文本-20201022-1.xlsx'
	pathexcel = r'D:/Work/TNT系统/算法测试素材/TNT-测试文本_备份2.xlsx'
	temp_zh_file = r'D:/Work/TNT系统/算法测试素材/temp_zh.txt'
	target_path = r'D:/Work/TNT系统/算法测试素材/'
	# url_zh = 'http://192.168.0.31:4444/event_extraction'
	url_zh = 'http://192.168.0.14:39999/event_extraction'
	url_en = 'http://192.168.0.14:4445/event_extraction_en'
	
	datas, lang_indies = excel(pathexcel)

	result = []
	temp = dict()
	for ind, lan_ind in enumerate(lang_indies):
		lan, number = lan_ind
		print(lan, number)
		if lan == '中文简体' or lan == '简体中文':
			url = ''
			# url = url_zh
			# temp[int(number)] = datas[ind]
		elif lan == '英语':
			url = url_en
			# url = ''
		else:
			url = ''
		if url:
			# print(datas[ind])
			res = post_event_extraction(int(number), url, {"content": [datas[ind]]})
			if res:
				result.extend(res)

	# 中文简体
	# res = format_list(result, 'zh')
	# write2excel(os.path.join(target_path, 'test_zh_2.xls'), res)
	
	# 英语
	res = format_list(result, 'en')
	write2excel(os.path.join(target_path, 'test_en_2.xls'), res)
	
	# for num, content in temp.items():
	# 	name = '{0}.txt'.format(num)
	# 	with open(os.path.join(target_path, name), 'w', encoding='utf-8') as fw:
	# 		fw.write(content)
	
	# test 2
	# p_path = os.path.join(target_path, '1.txt')
	# content = ''
	# with open(p_path, 'r', encoding='utf-8') as fr:
	#
	# post_event_extraction(1, url_zh, {"content": content})
	
	# test 3 单个测试
	# for ind, lan_ind in enumerate(lang_indies):
	# 	lan, number = lan_ind
	# 	print(lan, number)
	# 	if lan == '中文简体' or lan == '简体中文':
	# 		post_event_extraction(int(number), url_zh, {"content": [datas[ind]]})
	# 		# temps = split_txt(datas[ind])
	# 		# print(temps)
	# 		break

