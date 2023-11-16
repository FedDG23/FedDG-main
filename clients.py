import torch
import numpy as np
import logging
import copy


class Client:
    def __init__(self, client_id, generator=None, latents=None, f_latents=None, optimizer=None, criterion=None,
                 each_worker_data=None, each_worker_label=None, indices_cl_class=None, model=None, schedular=None,
                 args=None):
        self.client_id = client_id
        self.generator = generator
        self.latents = latents
        self.f_latents = f_latents
        self.optimizer = optimizer
        self.criterion = criterion
        self.each_worker_data = each_worker_data
        self.each_worker_label = each_worker_label
        self.indices_cl_class = indices_cl_class
        self.model = model
        self.schedular = schedular
        tmp_list = [(1 if len(indices_cl_class[c]) > 9 else 0) for c in range(len(indices_cl_class))]
        self.cpc_list = np.array([i*np.ones(args.ipc, dtype=np.int64) for i in tmp_list]).reshape(-1).astype(bool)
        self.class_list = np.array([c for c in range(len(indices_cl_class)) if len(indices_cl_class[c]) > 9])
        logging.info(str(client_id) + "  " + str(len(self.class_list)) + " " + str(self.class_list))

    def get_generator(self):
        return self.generator

    def set_generator(self, generator):
        self.generator = generator

    def get_G_param(self):
        tmp_params = [param.clone() for param in self.generator.parameters() if param.requires_grad != False]
        return torch.cat([xx.reshape((-1)) for xx in tmp_params], dim=0).squeeze(0)

    def set_G_param(self, agg_para):
        idx = 0
        for j, (param) in enumerate(self.generator.named_parameters()):
            if param[1].requires_grad:
                tmp_param = agg_para[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                param[1].data = tmp_param
                idx += param[1].nelement()

    def train_net(self, local_round, batch_size, device, num_batches=0, is_mtt=False, args=None):
        loss_num = 0.0
        cnt = 0
        if num_batches == 0: num_batches = (len(self.each_worker_data) // batch_size) + 10
        starting_params = [p.detach().clone().cpu() for p in self.model.parameters()]
        start_epoch = args.max_start_epoch
        for k in range(local_round):
            random_order = np.random.permutation(len(self.each_worker_data))
            for j in range((len(self.each_worker_data) - 1) // batch_size + 1):
                if j >= num_batches:
                    break
                data = self.each_worker_data[random_order[j * batch_size:
                                                min(len(self.each_worker_data), (j + 1) * batch_size)]].to(device)
                label = self.each_worker_label[random_order[j * batch_size:
                                                min(len(self.each_worker_data), (j + 1) * batch_size)]].to(device)
                self.model.to(device)
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss_num += loss.item()
                cnt += 1
                loss.backward()
                self.optimizer.step()
                if self.schedular is not None:
                    self.schedular.step()
        target_params = [p.detach().clone().cpu() for p in self.model.parameters()]
        return loss_num / cnt, starting_params, target_params, start_epoch

    def train_net_fedprox(self, local_round, batch_size, device, num_batches=0, global_net=None, args=None):
        loss_num = 0.0
        cnt = 0
        if num_batches == 0: num_batches = (len(self.each_worker_data) // batch_size) + 10
        global_weight_collector = list(global_net.to(device).parameters())
        for k in range(local_round):
            random_order = np.random.permutation(len(self.each_worker_data))
            for j in range((len(self.each_worker_data) - 1) // batch_size + 1):
                if j >= num_batches:
                    break
                data = self.each_worker_data[random_order[j * batch_size:
                                                min(len(self.each_worker_data), (j + 1) * batch_size)]].to(device)
                label = self.each_worker_label[random_order[j * batch_size:
                                                min(len(self.each_worker_data), (j + 1) * batch_size)]].to(device)
                self.model.to(device)
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)

                # for fedprox
                fed_prox_reg = 0.0
                mu = 0.001
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss_num += loss.item()
                cnt += 1
                loss.backward()
                self.optimizer.step()
                if self.schedular is not None:
                    self.schedular.step()
        return loss_num / cnt

    def train_net_fednova(self, local_round, batch_size, device, num_batches=0, global_net=None, args=None):
        loss_num = 0.0
        cnt = 0
        rho = 0
        if num_batches == 0: num_batches = (len(self.each_worker_data) // batch_size) + 10
        for k in range(local_round):
            random_order = np.random.permutation(len(self.each_worker_data))
            for j in range((len(self.each_worker_data) - 1) // batch_size + 1):
                if j >= num_batches:
                    break
                data = self.each_worker_data[random_order[j * batch_size:
                                                min(len(self.each_worker_data), (j + 1) * batch_size)]].to(device)
                label = self.each_worker_label[random_order[j * batch_size:
                                                min(len(self.each_worker_data), (j + 1) * batch_size)]].to(device)
                self.model.to(device)
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss_num += loss.item()
                cnt += 1
                loss.backward()
                self.optimizer.step()
                if self.schedular is not None:
                    self.schedular.step()
        global_net.to(device)
        a_i = (cnt - rho * (1 - pow(rho, cnt)) / (1 - rho)) / (1 - rho)
        global_net.to(device)
        global_model_para = global_net.state_dict()
        net_para = self.model.state_dict()
        norm_grad = copy.deepcopy(global_net.state_dict())
        for key in norm_grad:
            norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)
        return loss_num / cnt, a_i, norm_grad, (len(self.each_worker_data) - 1) // batch_size + 1

