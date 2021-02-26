import json
import time
from urllib import request as R


# res是个list
def send(cl):
    url = 'http://192.168.0.14:5464/wiki_predict'
    od = {'content': cl, 'model': '0702_svm'}
    jd = json.dumps(od).encode(encoding='utf-8')
    header_dict = {'Content-Type': 'application/json'}
    req = R.Request(url=url, data=jd, headers=header_dict)
    res = R.urlopen(req)
    res = res.read().decode('utf-8')
    res = json.loads(res)
    return res


cd = {}


def acc_jug(rl):
    right = 0
    for r in rl:
        tp = r['class']
        if tp == '人物':
            right += 1
        else:
            if tp in cd.keys():
                cd[tp] += 1
            else:
                cd[tp] = 1
    acc = float(right)/float(len(rl))
    return acc


def test():
    classified = []
    temp_list = []
    total_time = 0
    with open('D:/工作2020/7.16/person_zh', 'r', encoding='utf-8') as fr:
        c = 0
        l = 0
        for line in fr:
            sj = json.loads(line)
            temp_list.append(sj)
            c += 1
            if c >= 100:
                c = 0
                l += 1
                print(l)
                b_time = int(time.time() * 1000 * 1000)
                cr = send(temp_list)
                a_time = int(time.time() * 1000 * 1000)
                total_time += (a_time-b_time)
                classified.extend(cr)
                temp_list = []
            # if l == 20:
            #     break
    cr = send(temp_list)
    classified.extend(cr)
    accuracy = acc_jug(classified)
    print(accuracy)
    avg_time = float(total_time)/float(len(classified))
    print(avg_time)
    print(cd)


if __name__ == '__main__':
    test()
