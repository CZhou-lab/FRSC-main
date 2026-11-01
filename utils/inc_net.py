import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinear, CosineLinear_demo, AC_Linear, NN_Linear
import timm
import numpy as np
from torch.nn import functional as F
import math

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=args["nb_classes"])
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=args["nb_classes"])
        model.out_dim = 768
        return model.eval()    
    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        ffn_adapter_scalar = args["scalar"]
        if args["model_name"] == "frsc":
            from backbone import vit_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar= ffn_adapter_scalar,
                ffn_num=ffn_num,
                d_model=768,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_adapter":
                model = vit_adapter.vit_base_patch16_224_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_adapter":
                model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()       
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class MyNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.proxy_fc = None
        self.prototype = None
        self.prototype_cur = None
        self.low_rank = 16
        # self.W1 = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))
        # self.W2 = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))
        # self.W = nn.Parameter(0.1 * torch.randn(self.out_dim, self.out_dim))
        self.H = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)

    def update_H(self):
        # self.W1 = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))
        # self.W2 = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))
        # self.W = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))
        # self.W = nn.Parameter(0.1 * torch.randn(self.out_dim, self.out_dim))
        self.H = nn.Parameter(0.1 * torch.randn(self.out_dim, self.low_rank))

    def update_fc(self, nb_classes):
        self._cur_task += 1
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, :] = nn.Parameter(weight)

        del self.fc
        self.fc = fc
        del fc
        for param in self.fc.parameters():
            param.requires_grad = False

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def generate_fc_nn(self, in_dim, out_dim):
        fc = NN_Linear(in_dim, out_dim)
        return fc
    
    def generate_demo_fc(self, in_dim, out_dim):
        fc = CosineLinear_demo(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.forward_proto(x)["add"]

    def forward_proto(self, x, task=0):
        output = {}
        x_adapter = self.backbone.forward_test(x)
        output.update({"adapter": x_adapter})
        return output
    
    def forward(self, x, test=False, task = 0):
        
        if not test:
            x_adapter = self.backbone.forward_train(x)
            out = self.proxy_fc(x_adapter)
        else:
            x_adapter = self.backbone.forward_test(x)
            
            out = self.compute_logits_center_cos(x_adapter)
            out.update({"adapter": x_adapter})
        return out

    def compute_logits(self, x_adapter):
        x_adapter = F.normalize(x_adapter, p=2, dim=-1)
        if len(x_adapter.shape) == 2:
            out = F.linear(F.normalize(x_adapter, p=2, dim=1), F.normalize(torch.Tensor(np.array(self.prototype)).to(self._device), p=2, dim=1))
        else:
            for i in range(self._cur_task + 1):
                if i == 0:
                    temp_proto = torch.Tensor(np.array(self.prototype[:self.init_cls])).to(self._device)
                    temp_proto = F.normalize(temp_proto, p=2, dim=-1)
                    # 先扩展维度对齐
                    temp_proto_exp = temp_proto.unsqueeze(1).unsqueeze(2)          # (B, 1, 1, N)
                    x_adapter_exp = x_adapter.unsqueeze(0)  # (1, C, 2, N)
                    sim = F.cosine_similarity(temp_proto_exp, x_adapter_exp, dim=-1)
                else:
                    temp_proto = torch.Tensor(np.array(self.prototype[self.init_cls + (i-1) * self.inc: self.init_cls + i * self.inc])).to(self._device)
                    temp_proto = F.normalize(temp_proto, p=2, dim=-1)
                    t = temp_proto.shape[1]
                    temp_proto_exp = temp_proto.unsqueeze(0)
                    x_adapter_exp = x_adapter[:,:t,:].unsqueeze(1)
                sim = F.cosine_similarity(temp_proto_exp, x_adapter_exp, dim=-1)  # (B, C, 2)

                # 2. 对两个 prototype 取平均 (B, C)
                sim_mean = sim.mean(dim=-1)
                if i == 0:
                    out = sim_mean.T
                else:
                    out = torch.cat((out, sim_mean), dim=1)
        
        return {"logits": out}
    
    def compute_logits_center(self, x_adapter):
        if len(x_adapter.shape) == 2:
            out = F.linear(F.normalize(x_adapter, p=2, dim=1), F.normalize(torch.Tensor(np.array(self.prototype)).to(self._device), p=2, dim=1))
        else:
            x_adapter = x_adapter.mean(dim=1)
            x_adapter = F.normalize(x_adapter, p=2, dim=-1)
            for i in range(self._cur_task + 1):
                if i == 0:
                    temp_proto = torch.Tensor(np.array(self.prototype[:self.init_cls])).to(self._device)
                    temp_proto = F.normalize(temp_proto, p=2, dim=-1)
                else:
                    temp_proto = torch.Tensor(np.array(self.prototype[self.init_cls + (i-1) * self.inc: self.init_cls + i * self.inc])).to(self._device)
                    temp_proto = temp_proto.mean(dim=1)
                    temp_proto = F.normalize(temp_proto, p=2, dim=-1)
                
                sim = x_adapter @ temp_proto.T
                if i == 0:
                    out = sim
                else:
                    out = torch.cat((out, sim), dim=1)
        
        return {"logits": out}
    
    def compute_logits_center_cos(self, x_adapter):
        # x_adapter = F.normalize(x_adapter, p=2, dim=-1)
        self.prototype = self.fc.weight.data
        prototype = self.prototype
        if len(x_adapter.shape) == 2:
            # out = self.fc(x_adapter)
            out = F.linear(F.normalize(x_adapter, p=2, dim=1), F.normalize(prototype, p=2, dim=1))
        else:
            prototype = F.normalize(prototype, p=2, dim=-1)
            x_adapter_norm = F.normalize(x_adapter, p=2, dim=-1, eps=1e-8)
            logits = []
            for i in range(self._cur_task + 1):
                if i == 0:
                    x_adapter_temp = x_adapter[:, 0, :]
                else:
                    input_adapter = x_adapter[:, 0:i+1, :]
                    sim = (x_adapter_norm[:, i:i+1, :] * x_adapter_norm[:, 0:i+1, :]).sum(dim=-1)
                    sim[:, i] = 0
                    sim = sim / (sim.sum(dim=1, keepdim=True) + 1e-6)
                    sim[:, i] = 1.0
                    x_adapter_temp = (input_adapter * sim.unsqueeze(-1)).sum(dim=1)
                x_adapter_temp = F.normalize(x_adapter_temp, p=2, dim=-1)
                # out = self.fc(x_adapter_temp)
                if i == 0:
                    temp_proto = prototype[:self.init_cls]
                    # temp_proto = temp_proto @ self.W @ self.W.T
                    # temp_proto = temp_proto @ self.W @ self.W.T
                else:
                    temp_proto = prototype[self.init_cls + (i-1) * self.inc: self.init_cls + i * self.inc]
                    # temp_proto = temp_proto @ self.W @ self.W.T
                    # temp_proto = self.compute_center(temp_proto, i)
                    # temp_proto = temp_proto @ self.W @ self.W.T
                
                logits.append(x_adapter_temp @ temp_proto.T)
            out = torch.cat(logits, dim=1)
        
        return {"logits": out}
        # return out
    
    def compute_center(self, input, t):
        center_vec = input[:, t, :]
        sim = F.cosine_similarity(center_vec.unsqueeze(1), input, dim=-1)
        sim[:, t] = 0
        # print(sim)
        # sim[sim < 0.5] = 0
        sim = sim / (sim.sum(dim=1, keepdim=True) + 1e-6)
        sim[:, t] = 1.0
        # print(sim)
        weights = sim.unsqueeze(-1)
        centroid = (input * weights).sum(dim=1)
        # centroid = input.mean(1)
        return centroid
    
    def compute_logits_single(self, x_adapter):

        if len(x_adapter.shape) == 2:
            x_adapter = F.normalize(x_adapter, p=2, dim=-1)
            out = F.linear(F.normalize(x_adapter, p=2, dim=1), F.normalize(torch.Tensor(np.array(self.prototype)).to(self._device), p=2, dim=1))
        else:
            x_adapter = x_adapter[:, -1, :]
            x_adapter = F.normalize(x_adapter, p=2, dim=-1)
            for i in range(self._cur_task + 1):
                if i == 0:
                    temp_proto = torch.Tensor(np.array(self.prototype[:self.init_cls])).to(self._device)
                    temp_proto = F.normalize(temp_proto, p=2, dim=-1)
                    sim = x_adapter @ temp_proto.T
                else:
                    temp_proto = torch.Tensor(np.array(self.prototype[self.init_cls + (i-1) * self.inc: self.init_cls + i * self.inc])).to(self._device)
                    temp_proto = temp_proto[:, -1, :]
                    temp_proto = F.normalize(temp_proto, p=2, dim=-1)
                    sim = x_adapter @ temp_proto.T

                if i == 0:
                    out = sim
                else:
                    out = torch.cat((out, sim), dim=1)
        
        return {"logits": out}

    def compute_simi(self, x_vit, x_adapter):
        x_adapter_norm = F.normalize(x_adapter, p=2, dim=-1)
        x_vit_norm = F.normalize(x_vit, p=2, dim=-1).unsqueeze(1)
        cos_simi = torch.sum(x_adapter_norm * x_vit_norm, dim=-1)
        cos_simi = F.softmax(cos_simi, dim=-1).unsqueeze(-1)
        output = torch.sum(cos_simi * x_adapter, dim=1)
        return output
    
