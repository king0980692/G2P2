import numpy as np

edges = []
with open('./cora/citations', 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        edges.append([])
        line = line.strip('\n').split()
        edges[i].append(line[0])
        edges[i].append(line[1])
        i += 1

edges = np.array(edges, dtype=int)
print('original edges number', edges.shape[0])
# print(edges[:50])
#
edges_uniq = np.unique(edges)
#
# print(edges_uniq[:50])
#
print('edges_uniq length', edges_uniq.shape[0])

id_list = []
tit_list = []
abs_list = []
lab_list = []
with open('./cora/id_tit_abs_lab.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        id = int(line[0])
        id_list.append(id)
        tit_list.append(line[1])
        abs_list.append(line[2])
        lab_list.append(line[3])

id_set = set(id_list)
print('number of nodes having text', len(id_list))

true_edges = []
for i in range(edges.shape[0]):
    if (edges[i][0] in id_set) & (edges[i][1] in id_set):
        true_edges.append(edges[i])

print('number of true_edges', len(true_edges))

true_edges_set = np.unique(np.array(true_edges)).tolist()
print(true_edges_set[:100])
true_edges_set = set(true_edges_set)

print('number of nodes in edges:', len(true_edges_set))


true_id_list = []
true_tit_list = []
true_abs_list = []
true_lab_list = []
for i in range(len(id_list)):
    if id_list[i] in true_edges_set:
        true_id_list.append(id_list[i])
        # tit = tit_list[i]
        tit = tit_list[i].lower()
        tit = tit.split()[1:]
        tit = ' '.join(tit)
        true_tit_list.append(tit)
        abs = abs_list[i].lower()
        abs = abs.split()[1:]
        abs = ' '.join(abs)
        true_abs_list.append(abs)
        if lab_list[i] == 'nan':
            lab = 'nan'
        else:
            lab = lab_list[i].replace('_', ' ').lower().strip('/').split('/')
            lab = ', '.join(lab)
        true_lab_list.append(lab)


id_map = {true_id_list[i]: i for i in range(len(true_id_list))}
# new_id_list = np.arange(len(true_id_list)).tolist()

print('true_tit_list', true_tit_list[:50])

print('true_abs_list', true_abs_list[:10])

test = true_abs_list[0]
print(test)
test = test.split('. ')
print(test)


with open('./cora/mapped_edges.txt', 'w') as f:
    for i in range(len(true_edges)):
        f.write(str(id_map[true_edges[i][0]]))
        f.write(' ')
        f.write(str(id_map[true_edges[i][1]]))
        f.write('\n')


with open('./cora/id_tit_abs_lab_final.txt', 'w') as f:
    for i in range(len(true_id_list)):
        f.write(str(i))
        f.write('\t')
        f.write(true_tit_list[i])
        f.write('\t')
        f.write(true_abs_list[i])
        f.write('\t')
        f.write(true_lab_list[i])
        f.write('\n')