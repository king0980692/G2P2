import numpy as np
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents




id_list = []
tit_list = []
abs_list = []
lab_list = []
lenth_list = []


with open('./cora/id_tit_abs_lab_final.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        line = line.strip().split('\t')
        id_list.append(line[0])
        # tit = preprocess_string(line[1])
        # tit = ' '.join(tit)
        tit_list.append(line[1])
        # abs = preprocess_string(line[2])
        # abs = ' '.join(abs)
        abs_list.append(line[2])
        abs = line[2].split()
        lenth_list.append(len(abs))
        lab_list.append(line[3])

print('number of node texts', len(lenth_list))  # 905453*0.8
# print()
lenth_list.sort()
print('0.7 lenth=', lenth_list[int(len(lenth_list) * 0.7)])
print('0.8 lenth=', lenth_list[int(len(lenth_list) * 0.8)])
print('0.9 lenth=', lenth_list[int(len(lenth_list)*0.9)])
print('average context length', round(np.mean(lenth_list), 2))

# with open('./cora/id_tit_abs_lab_final.txt', 'r') as f:
#     lines = f.readlines()
#     i = 0
#     for line in lines:
#         line = line.strip().split('\t')
#         id_list.append(line[0])
#         tit = preprocess_string(line[1])
#         tit_list.append(line[1])
#         abs = preprocess_string(line[2])
#         abs_list.append(line[2])
#         lab_list.append(line[3])



print('tit_list', tit_list[:50])
print('abs_list', abs_list[:50])

with open('./cora/train_text.txt', 'w') as f:
    for i in range(len(id_list)):
        f.write(id_list[i])
        f.write('\t')
        f.write(tit_list[i])
        f.write('\t')
        f.write(abs_list[i])
        f.write('\t')
        f.write(lab_list[i])
        f.write('\n')


print('done')