import torch
import torch.nn as nn
import torch.nn.functional as F

# second order
class line(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay, loss_fn):
        super().__init__()

        self.dim = dim
        self.user_size = user_size
        self.item_size = item_size
        self.size = user_size + item_size
        self.vertext_emb = nn.Embedding(self.size, dim)
        self.context_emb = nn.Embedding(self.size, dim)

        self.vertext_emb.weight.data = self.vertext_emb.weight.data.uniform_(-0.5, 0.5) / dim
        self.context_emb.weight.data = self.context_emb.weight.data.uniform_(-0.5, 0.5) / dim

    def save_embedding(self, rv_user_mapping, rv_item_mapping, saved_path):

        print(f"\n\nSaving Embedding to {saved_path}")
        vertex_emb = self.vertext_emb.weight.data

        output = []
        for _i in range(0, self.user_size):
            u_id = str(rv_user_mapping[_i])
            u_vec = vertex_emb[_i].tolist()
            vec_str = " ".join([ str(_v) for _v in u_vec ])

            output.append(u_id+"\t"+vec_str+"\n")

        for _i in range(self.user_size, vertex_emb.shape[0]): 
            i_id = str(rv_item_mapping[_i-self.user_size])
            i_vec = vertex_emb[_i].tolist()
            vec_str = " ".join([ str(_v) for _v in i_vec ])

            output.append(i_id+"\t"+vec_str+"\n")

        with open(saved_path, 'w') as f:
            f.writelines(output)
        
    def forward(self, u, i, js):
        """
        Args:
            u   (torch.LongTensor): tensor stored vertex(user or item) indexes. [batch_size,]
            i   (torch.LongTensor): tensor stored vertex(user or item) indexes which is prefered by vertex. [batch_size,]
            js  (torch.LongTensor): tensor stored vertexs(user or item) indexes which are not prefered by vertex. [batch_size,] * NEG
        """

        ## positive sample
        v1 = self.vertext_emb(u) # batch size * dim
        v2 = self.context_emb(i) # batch size * dim
        negs = -self.context_emb(js)

        mulpositivebatch = torch.mul(v1, v2)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        mulnegativebatch = torch.mul(v1.view(len(v1), 1, self.dim), negs)
        negtivebatch = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch, dim=2)), dim=1)

        loss = positivebatch + negtivebatch
        return -torch.mean(loss)

        
