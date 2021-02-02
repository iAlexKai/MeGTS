import torch
from fairseq.models.bart.model import BARTModel

bpe_json = '../data_process/poet/resources/encoder.json'
bart = BARTModel.from_pretrained(
    '../checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../data-bin/poet',
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32


def _load_dict(json_path):
    import json
    with open(json_path, 'r') as file:
        ind2char = {}
        word_dict = json.loads(file.read())
        for (key, val) in word_dict.items():
            ind2char[val] = key
        return ind2char


ind2char = _load_dict(bpe_json)

with open('../data_process/poet/test.source') as source, open('../output/poet/test.hypo', 'w') as fout:

    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                # import pdb
                # pdb.set_trace()
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=1.0, max_len_b=30, min_len=5, no_repeat_ngram_size=3)
                # import pdb
                # pdb.set_trace()
            for hypothesis in hypotheses_batch:
                fout.write("".join(ind2char[i] for i in hypothesis) + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    # import pdb
    # pdb.set_trace()
    # 剩下不足一个batch的做sample
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=1.0, max_len_b=30, min_len=5, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write("".join(ind2char[i] for i in hypothesis) + '\n')
            fout.flush()