import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MyNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from sklearn.decomposition import PCA
from torch.distributions.multivariate_normal import MultivariateNormal
from backbone.linears import CosineLinear
import copy
import os
from sklearn.manifold import TSNE
import time

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MyNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.lambda1 = args["lambda1"]
        self.lambda2 = args["lambda2"]
        self.pa_lr = args["pa_lr"]
        self.pa_epochs = args["pa_epochs"]
        self.size = 224
        self.proto_add_all = []
        self.proto_vit_all = []
        self.proto_adapter_all = []
        self.labels = []
        self.proto_add = []
        self.proto_vit = []
        self.proto_adapter = []
        self.add_cov = []
        self.add_cov_pre = []
        self.mean_proto = []
        self.temp_project_before_mean = args.get("temp_project_before_mean", False)
        if not isinstance(self.temp_project_before_mean, bool):
            raise ValueError("temp_project_before_mean must be a boolean")
        # Frozen H factors, in the same order as backbone.adapter_list.
        self._adapter_h_history = []

    def after_task(self):
        self._known_classes = self._total_classes
        if self.temp_project_before_mean:
            # update_H() resets H, so retain this adapter's final value first.
            self._adapter_h_history.append(self._network.H.detach().cpu().clone())
        self._network.backbone.add_adapter_to_list()
        self._network.update_H()
        # if self._cur_task >= 1:
        #     self.tsne(False, True)
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls

    def cat_proto(self, proto_pre, proto_now):
        if len(self.add_cov_pre[0].shape) == 2:
            proto_pre = torch.Tensor(np.array(proto_pre)).unsqueeze(1)
            proto_now = torch.Tensor(np.array(proto_now)).unsqueeze(1)
        else:
            proto_pre = torch.Tensor(np.array(proto_pre))
            proto_now = torch.Tensor(np.array(proto_now)).unsqueeze(1)
        proto = torch.cat((proto_pre, proto_now), dim=1)
        
        return list(proto)

    def replace_fc(self, train_loader):
        
        proto_adapter, proto_cov, class_list = self.cal_proto(train_loader)

        if self._cur_task == 0:
            self.proto_adapter_all = proto_adapter
            self.add_cov = proto_cov
            self.labels = class_list
            self._network.fc.weight.data[self._known_classes:self._total_classes, :] = torch.Tensor(np.array(proto_adapter))
            self._network.prototype_cur = self._network.fc.weight.data
            self._network.prototype = self._network.fc.weight.data
        else:
            self._network.prototype_cur = torch.cat((self._network.prototype_cur, torch.Tensor(np.array(proto_adapter)).to(self._device)), dim=0)
            proto_cov = self.cat_proto(self.add_cov_pre, proto_cov)
            proto_adapter = self.cat_proto(self.proto_adapter_all_pre, proto_adapter)

            self.labels = torch.cat((self.labels, class_list), dim=0)
            self.add_cov.extend(proto_cov)
            self.proto_adapter_all.extend(proto_adapter)
            proto_adapter = self._network.compute_center(torch.Tensor(np.array(proto_adapter)), self._cur_task)
            self._network.fc.weight.data[self._known_classes:self._total_classes, :] = proto_adapter
            self._network.prototype = self._network.fc.weight.data

    def cal_proto(self, train_loader):
        model = self._network.to(self._device)
        model = model.eval()
        
        with torch.no_grad():
            embedding_list_add, embedding_list_vit, embedding_list_adapter, label_list = [], [], [], []
            for i, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone.forward_cur(data)

                embedding_list_adapter.append(embedding.cpu())
                label_list.append(label.cpu())

            embedding_list_adapter = torch.cat(embedding_list_adapter, dim=0)
            label_list = torch.cat(label_list, dim=0)

            class_list = np.unique(self.train_dataset_for_protonet.labels)
            proto_adapter, proto_cov = [], []
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding_adapter = embedding_list_adapter[data_index]

                adapter_mean = embedding_adapter.mean(0)
                if len(embedding_adapter.shape) == 2:
                    proto_cov.append(torch.cov(embedding_adapter.T) + (torch.eye(embedding_adapter.shape[-1]) * 1e-3))
                else:
                    x_centered = embedding_adapter - adapter_mean.unsqueeze(0)
                    proto_cov.append(torch.matmul(x_centered.permute(1, 2, 0), x_centered.permute(1, 0, 2)) / (x_centered.shape[0] - 1))

                proto_adapter.append(adapter_mean)
        label = torch.Tensor(np.array(class_list))
        return proto_adapter, proto_cov, label
    
    def cal_proto_pre(self, train_loader):
        model = self._network.to(self._device)
        model = model.eval()
        
        with torch.no_grad():
            embedding_list_adapter, label_list = [], []
            for i, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)

                embedding = model.backbone.forward_old(data)

                embedding_list_adapter.append(embedding.cpu())
                label_list.append(label.cpu())

            embedding_list_adapter = torch.cat(embedding_list_adapter, dim=0)
            label_list = torch.cat(label_list, dim=0)

            class_list = np.unique(self.train_dataset_for_protonet.labels)
            proto_adapter, proto_cov = [], []
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding_adapter = embedding_list_adapter[data_index]
                adapter_mean = embedding_adapter.mean(0)
                if len(embedding_adapter.shape) == 2:
                    proto_cov.append(torch.cov(embedding_adapter.T) + (torch.eye(embedding_adapter.shape[-1]) * 1e-3))
                else:
                    x_centered = embedding_adapter - adapter_mean.unsqueeze(0)
                    proto_cov.append(torch.matmul(x_centered.permute(1, 2, 0), x_centered.permute(1, 0, 2)) / (x_centered.shape[0] - 1))

                proto_adapter.append(adapter_mean)

        self.proto_adapter_all_pre = proto_adapter
        self.add_cov_pre = proto_cov

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        # before train
        self._network.update_fc(self._total_classes)
        # self._network.update_H()
        
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        if self._cur_task > 0:
            self.cal_proto_pre(self.train_loader)
        
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        self.replace_fc(self.train_loader_for_protonet)
        ############## prototype align ##############
        if self._cur_task > 0:
            self.prototype_align()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0:
            optimizer = self.get_optimizer(model=self._network, lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
           
            optimizer = self.get_optimizer(model=self._network, lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)
        
    
    def get_optimizer(self, model, lr):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': lr, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params, 
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )
        return optimizer


    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    @torch.no_grad()
    def _compute_experience_temp(self):
        prototypes = torch.Tensor(np.array(self.proto_adapter_all_pre)).to(self._device)
        if not self.temp_project_before_mean:
            return prototypes.mean(1) if prototypes.ndim == 3 else prototypes

        # forward_old() omits the adapter dimension when there is only one.
        if prototypes.ndim == 2:
            prototypes = prototypes.unsqueeze(1)
        if prototypes.ndim != 3 or prototypes.shape[1] != len(self._adapter_h_history):
            raise ValueError("Historical prototypes and saved adapter H factors must match")

        projected = []
        for adapter_id, historical_h in enumerate(self._adapter_h_history):
            historical_h = historical_h.to(device=prototypes.device, dtype=prototypes.dtype)
            projected.append(prototypes[:, adapter_id] @ historical_h @ historical_h.T)
        # Normalize only after averaging, as in the original training path.
        return torch.stack(projected, dim=1).mean(1)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        
        if self._cur_task == 0:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']
            experience_temp = self._compute_experience_temp()
            identity_w = torch.eye(
                self._network.H.shape[0],
                device=self._device,
                dtype=self._network.H.dtype,
            )
        
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = torch.where(
                    targets - self._known_classes >= 0,
                    targets - self._known_classes,
                    -1,
                )
                output = self._network.backbone.forward_train(inputs)
                
                if self._cur_task == 0:
                    logits = self._network.proxy_fc(output)["logits"]
                    loss_ce = F.cross_entropy(logits, aux_targets)
                    loss = loss_ce 
                else:
                    temp = experience_temp[aux_targets.cpu()]
                    temp = F.normalize(temp, p=2, dim=-1)

                    output = F.normalize(output, p=2, dim=-1)

                    logits = self._network.proxy_fc(output)["logits"]
                    
                    projection = self._network.H @ self._network.H.T
                    loss_orth = F.mse_loss(
                        projection @ projection.T,
                        identity_w,
                        reduction="mean",
                    )
                    loss_cos = F.mse_loss(output, temp @ self._network.H @ self._network.H.T, reduction="mean")

                    loss_ce = F.cross_entropy(logits, aux_targets)
                    
                    loss = loss_ce + self.lambda1 * loss_cos + self.lambda2 * loss_orth
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)
            if scheduler:
                scheduler.step()
            train_acc = float(np.around(tensor2numpy(correct) * 100 / total, decimals=2))

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        if epochs != 0:
            logging.info(info)

    def prototype_align(self):
        fc = CosineLinear(self._network.out_dim, self._total_classes)
        fc.weight.data = nn.Parameter(copy.deepcopy(self._network.fc.weight.data))
        fc.to(self._device)
        fc.train()
        run_epochs = self.pa_epochs
        optimizer = optim.Adam([{'params': fc.parameters()}], lr=self.pa_lr, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
        num_sampled_pcls = self.batch_size * 5
        sampled_data, sampled_label = self.sample_features(num_sampled_pcls)
        
        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = fc(inp)
                logits = outputs['logits'][:, :self._total_classes]
                loss = F.cross_entropy(logits, tgt)
                
                _, preds = torch.max(logits, dim=1)
                
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                run_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)
        self._network.fc.weight.data = copy.deepcopy(fc.weight.data)

        logging.info(info)
    
    def sample_features(self, num_sampled_pcls):
        sampled_data = []
        sampled_label = []

        for class_idx in range(self._total_classes):
            mean = self.proto_adapter_all[class_idx]
            cov = self.add_cov[class_idx]
            if len(mean.shape) == 1:
                m = MultivariateNormal(mean.float(), cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                sampled_data_single = []
                for i in range(mean.shape[0]):
                    mean_temp = mean[i, :]
                    cov_temp = cov[i, :, :] + 1e-3 * torch.eye(cov.shape[-1])
                    m = MultivariateNormal(mean_temp.float(), cov_temp.float())
                    sampled_data_single_pre = m.sample(sample_shape=(num_sampled_pcls, ))
                    sampled_data_single.append(sampled_data_single_pre)
                
                sampled_data_single = torch.stack(sampled_data_single, dim=1)
                sampled_data_single = self._network.compute_center(sampled_data_single, mean.shape[0] - 1)
                sampled_data.append(sampled_data_single)
                sampled_label.extend([class_idx] * num_sampled_pcls)

        sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
        sampled_label = torch.tensor(sampled_label).long().to(self._device)
        return sampled_data, sampled_label
    
    
    def eval_task(self):
        total_time = 0.0
        num = len(self.test_loader)
        start_time = time.time()
        y_pred, y_true = self._eval_cnn(self.test_loader)
        end_time = time.time()
        total_time = (end_time - start_time) / num
        logging.info("inference time: {:8.4f}".format(total_time / num))
        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None

        return cnn_accy, nme_accy

    def eval_task_new(self):

        if self._cur_task == 0:
            self.mean_proto = self._network.fc.weight.data[:self.init_cls,:].mean(dim=0).unsqueeze(0)
        else:
            temp = self._network.fc.weight.data[self._total_classes - self.inc:self._total_classes,:].mean(dim=0).unsqueeze(0)
            self.mean_proto = torch.cat((self.mean_proto, temp), dim=0)
        
        if self._cur_task == 0:
            y_pred, y_true = self._eval_cnn(self.test_loader)
            cnn_accy = self._evaluate(y_pred, y_true)
        else:
            y_pred, y_true, simi = self._eval_cnn_new(self.test_loader)

            y_pred = torch.topk(y_pred, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred = y_pred.numpy()
            cnn_accy = self._evaluate(y_pred, y_true)


        nme_accy = None
        return cnn_accy, nme_accy

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return float(np.around(tensor2numpy(correct) * 100 / total, decimals=2))

    def _eval_cnn_new(self, loader):

        calc_task_acc = True
        
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        self._network.eval()
        y_pred, y_true, simi = [], [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)
            fea = outputs["add"]
            out = F.linear(F.normalize(fea, p=2, dim=1), F.normalize(self.mean_proto, p=2, dim=1))
            out = F.softmax(out, dim=1)
            out = out.repeat_interleave(self.init_cls, dim=1)
            simi.append(out.cpu().numpy())
            predicts = outputs["logits"]
            # predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs["logits"].shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs["logits"][i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs["logits"], dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
                
        return torch.from_numpy(np.concatenate(y_pred)), np.concatenate(y_true), torch.from_numpy(np.concatenate(simi))  # [N, topk]

    def _eval_cnn(self, loader):
        calc_task_acc = True
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True, task=self._cur_task)["logits"][:, :self._total_classes]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)
        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
                
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

