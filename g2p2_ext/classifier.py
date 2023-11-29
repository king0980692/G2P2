
import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, args, label_params):
        super(Classifier, self).__init__()

        self.vars = nn.ParameterList()
        w1 = nn.Parameter(label_params)
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(args.num_labels)))


    def forward(self, x):
        vars = self.vars
        tx = F.linear(x, vars[0], vars[1])
        return torch.log_softmax(tx.squeeze(), dim=-1)
        # return torch.softmax(tx.squeeze(), dim=-1)

    def parameters(self):
        return self.vars


    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)
