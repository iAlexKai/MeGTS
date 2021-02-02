from collections import Counter
import json

# Read all files
items = ['train', 'valid', 'test']
item_lens = []
sources, targets = [], []

for item in items:
    with open('{}.txt'.format(item), 'r') as file:
        lines = file.read().strip().split('\n')
        item_lens.append(len(lines))
        for line in lines:
            content = line.split('\t')
            sources.append(content[0])
            targets.append(content[1])


# tokenize and generate dictionary
counter = Counter()
for line in sources:
    counter.update(line)
for line in targets:
    counter.update(line)

dictionary = []
for item in counter.most_common():
    dictionary.append(item[0])


chars_map = {k: v+4 for v, k in enumerate(dictionary)}

# counter_file = "dict.txt"
# with open(counter_file, 'w') as file:
#     for k, v in chars_map.items():
#         file.write("{} {}\n".format(v, 0))

chars_map.update({'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3})


def save_json(file_name, smap):
    with open(file_name, 'w') as file:
        file.write(json.dumps(smap, ensure_ascii=False))
# save_json("resources/encoder.json", chars_map)


# Write to file
start_id = 0
for i, item in enumerate(items):
    sources_to_write = sources[start_id: start_id + item_lens[i]]
    targets_to_write = targets[start_id: start_id + item_lens[i]]
    start_id += item_lens[i]
    with open('{}.source'.format(item), 'w') as output:
        for line in sources_to_write:
            index_list = [str(chars_map[item]) for item in line]
            output.write(' '.join(index_list) + '\n')

    with open('{}.target'.format(item), 'w') as output:
        for line in targets_to_write:
            index_list = [str(chars_map[item]) for item in line]
            output.write(' '.join(index_list) + '\n')

    with open('{}-source.origin'.format(item), 'w') as output:
        for line in sources_to_write:
            output.write(line + '\n')

    with open('{}-target.origin'.format(item), 'w') as output:
        for line in targets_to_write:
            output.write(line + '\n')



# import pdb
# pdb.set_trace()
# target_txt = open('{}-target.txt'.format(item), 'w')
# source_txt.close()
# target_txt.close()
