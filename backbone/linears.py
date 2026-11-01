'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy

class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=False))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=False)
        trunc_normal_(_fc.weight, std=.02)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False

        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(1*(F.linear(F.normalize(fc_inp, p=2, dim=1),F.normalize(self.heads[ti][0].weight, p=2, dim=1))))
        out = {'logits': torch.cat(out, dim=1)}
        return out

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):

        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        # out = F.linear(input, self.weight)
        # out = F.normalize(out, p=2, dim=1)
        # out = out - out.min(dim=1, keepdim=True)[0]
        # out = torch.nn.functional.softmax(out, dim=1)

        # out = torch.cdist(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), p=2)
        # out2 = torch.cdist(input, self.weight, p=2)
        # out = torch.argmin(dists, dim=1)
        # out = F.linear(input, self.weight)
        # out = out / torch.norm(out)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    
    def forward_reweight(self, input, cur_task, alpha=0.1, beta=0.0, init_cls=10, inc=10, out_dim=768, use_init_ptm=False):
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc
            
            out = 0.0
            for j in range((self.in_features // out_dim)):
                # PTM feature
                if use_init_ptm and j == 0:
                    input_ptm = F.normalize(input[:, 0:out_dim], p=2, dim=1)
                    weight_ptm = F.normalize(self.weight[start_cls:end_cls, 0:out_dim], p=2, dim=1)
                    out_ptm = beta * F.linear(input_ptm, weight_ptm)
                    out += out_ptm
                    continue

                input1 = F.normalize(input[:, j*out_dim:(j+1)*out_dim], p=2, dim=1)
                weight1 = F.normalize(self.weight[start_cls:end_cls, j*out_dim:(j+1)*out_dim], p=2, dim=1)
                if use_init_ptm:
                    if j != (i+1):
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)
                else:
                    if j != i:
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)

                out += out1
            
            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1) if i != 0 else out
                
        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all
        
        return {'logits': out_all}

class CosineLinear_demo(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear_demo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input, add_basis_all=None):
        L = add_basis_all.shape[0]
        out = torch.zeros(input.shape[0], self.weight.shape[0]).to(self.weight.device)
        for i in range(L):
            input_ = input @ add_basis_all[i, :, :]
            out_ = F.linear(F.normalize(input_, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
            out += out_

        out = out / L

        # out = F.linear(input, self.weight)
        # out = F.normalize(out, p=2, dim=1)
        # out = out - out.min(dim=1, keepdim=True)[0]
        # out = torch.nn.functional.softmax(out, dim=1)

        # out = torch.cdist(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), p=2)
        # out2 = torch.cdist(input, self.weight, p=2)
        # out = torch.argmin(dists, dim=1)
        # out = F.linear(input, self.weight)
        # out = out / torch.norm(out)
        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

class AC_Linear(nn.Module):
    def __init__(self, in_features, Hidden, out_dim):
        super(AC_Linear, self).__init__()
        self.in_features = in_features
        self.hidden = Hidden
        self.out_features = out_dim
        bias_fe = False

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.hidden, bias=bias_fe),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden, self.out_features, bias=False))
    def forward(self, input):
        hidden_feature = self.fc[:2](input)
        out = self.fc(input)

        return {'hidden_feature': hidden_feature, 'logits': out}
    
class NN_Linear(nn.Module):
    def __init__(self, in_features, out_dim):
        super(NN_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_dim
        bias_fe = True

        self.fc = nn.Linear(self.in_features, self.out_features, bias=bias_fe)
    
    def forward(self, input):

        out = self.fc(input)

        return {'logits': out}
    

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)
