import numpy as np



lab_list = []
with open('./cora/id_tit_abs_lab_final.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        line = line.strip().split('\t')
        lab_list.append(line[3])

lab_list, lab_count = np.unique(np.array(lab_list), return_counts=True)

print('lab_list', lab_list)
print('lab_count', lab_count)
print('lab num', lab_list.shape)

lab_list = lab_list.tolist()
print('lab_list', lab_list)

with open('./cora/lab_list.txt', 'w') as f:
    for lab in lab_list:
        f.write(lab)
        f.write('\t')

