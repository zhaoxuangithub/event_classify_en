import os
import sys
import json
import re

sys.path.append(os.path.abspath('..'))


def json_combine(text):
    data = json.loads(text)
    sumString = re.sub('\n', '', data['sumString'])
    if 'property' in data.keys():
        property_dic = data['property']
        plus_string = '|| '
        for k, v in property_dic.items():
            plus_string += k + ': ' + v + ' '
        combined_string = sumString + plus_string + '\n'
    else:
        combined_string = sumString + '\n'
    return combined_string


def dic_combine(data):
    sumString = re.sub('\n', '', data['sumString'])
    if 'property' in data.keys():
        property_dic = data['property']
        plus_string = '|| '
        for k, v in property_dic.items():
            plus_string += k + ': ' + v + ' '
        combined_string = sumString + plus_string + '\n'
    else:
        combined_string = sumString + '\n'
    return combined_string