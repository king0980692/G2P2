import numpy as np

f_list = []
l_list = []

with open('./cora/classifications', 'r') as f:
    lines = f.readlines()
    print(lines[-2:])
    i = -1
    for line in lines:
        i += 1
        if line == '\t\n':
            continue
        # print(line)
        line = line.strip().split()
        f_list.append(line[0])
        l_list.append(line[1])

f_l_map = {f_list[i]: l_list[i] for i in range(len(f_list))}

label_f_set = set(f_list)

id_list = []
f2_list = []
with open('./cora/id_file.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        id_list.append(line[0])
        f2_list.append(line[1])

i_f_map = {id_list[i]: f2_list[i] for i in range(len(id_list))}

true_id_list = []
title_list = []
abs_list = []
with open('./cora/id_tit_abs.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        true_id_list.append(line[0])
        title_list.append(line[1])
        abs_list.append(line[2])

label_list = []
real_index = []
for i in range(len(true_id_list)):
    if i_f_map[true_id_list[i]] in label_f_set:
        label = f_l_map[i_f_map[true_id_list[i]]]
        label_list.append(label)
        real_index.append(i)
    else:
        label_list.append('nan')
        # nan_index.append(i)

print('label_list', label_list[:50])
print('real index number', len(real_index))
print('true_id_list length', len(true_id_list))

with open('./cora/id_tit_abs_lab.txt', 'w') as f:
    for i in range(len(true_id_list)):
        f.write(true_id_list[i])
        f.write('\t')
        f.write(title_list[i])
        f.write('\t')
        f.write(abs_list[i])
        f.write('\t')
        f.write(label_list[i])
        f.write('\n')

# print(line)
# if i > 100:
#     break
