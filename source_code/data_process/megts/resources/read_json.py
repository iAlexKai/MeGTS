import json


def index_to_char(char2ind):
    ind2char = {}
    for (key, val) in char2ind.items():
        ind2char[val] = key
    return ind2char



with open('encoder.json', 'r') as file:
    content = file.read().strip()
    char2ind = json.loads(content)
    ind2char = index_to_char(char2ind)
    while True:
        index_list_str = str(input())

        index_list = eval(index_list_str)
        # import pdb
        # pdb.set_trace()
        print("".join(ind2char[i] for i in index_list))
        

        # index_list = index_list_str.split(' ')
        # print("".join(ind2char[int(i)] for i in index_list))









