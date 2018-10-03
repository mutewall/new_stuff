from __future__ import division
from __future__ import print_function
import random
import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
import math
import pdb
import gc


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def preprocess(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)

class BaseModel(nn.Module):
    def __init__(self, input_channel_size, num_class):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        #self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        #self.bn4 = nn.BatchNorm2d(32)
        # self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1, bias = False)
        # self.bn5 = nn.BatchNorm2d(128)
        self.dense1 = nn.Linear(64*4, 64)
        self.dense2 = nn.Linear(64, num_class)
        
        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight.data, 1.0)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        # out = self.bn5(self.conv5(out))       
        # out = F.relu(out)
        out = out.view(-1, 64*4)
        out = self.dense1(out)
        out = self.dense2(out)
        return F.log_softmax(out, dim=1) 

class NodeUapter(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        self.num_layers = num_layers
        for i in xrange(self.num_layers):
            self.add_module('lstm_layer')

class GenerateGradientLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GenerateGradientLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

        init.xavier_normal_(self.linear.weight.data)
        init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        return self.linear(x)

class Node(nn.Module):
    def __init__(self, node_id, updater_set, hidden_size, init=None):
        super(Node, self).__init__()
        self.node_id = node_id
        if init == None:
            self.h = torch.zeros((1, hidden_size)).to(device)
            self.c = torch.zeros((1, hidden_size)).to(device)
        else:
            self.h = init[0]
            self.c = init[1]

        for i, updater in enumerate(updater_set[0]):
            self.add_module('lstm_layer{}'.format(i), updater)
        for j, updater in enumerate(updater_set[1]):
            self.add_module('output_layer{}'.format(j), updater)
        self.num_lstm_layer = i + 1
        self.num_output_layer = j + 1

    def update(self, grad):
        for i in xrange(self.num_lstm_layer):
            self.h, self.c = self._modules['lstm_layer{}'.format(i)](grad, self.h, self.c)
            temp_grad = F.tanh(self._modules['output_layer{}'.format(0)](self.h))
            for j in xrange(1, self.num_output_layer - 1 ):
                temp_grad = F.tanh(self._modules['output_layer{}'.format(j)](temp_grad))
            grad = F.tanh(self._modules['output_layer{}'.format(self.num_output_layer-1)](temp_grad))
            grad = preprocess(grad)

    def output_grad(self):
        temp_grad = F.tanh(self._modules['output_layer{}'.format(0)](self.h))
        for j in xrange(1, self.num_output_layer - 1 ):
            temp_grad = F.tanh(self._modules['output_layer{}'.format(j)](temp_grad))
        grad = F.tanh(self._modules['output_layer{}'.format(self.num_output_layer-1)](temp_grad))
        return grad    

class EdgeUpdater(nn.Module):
    def __init__(self, input_size, middle_size = None, output_size = None):
        super(EdgeUpdater, self).__init__()        
        if output_size is not None:
            self.edge_update_func = nn.Linear(input_size, output_size)
            self.interpolation_func = nn.Linear(input_size, output_size)

        else:
            self.edge_update_func = nn.Linear(input_size, middle_size)
            self.interpolation_func = None

        init.xavier_normal_(self.edge_update_func.weight.data)
        init.constant_(self.edge_update_func.bias.data, 0)
        
        if self.interpolation_func is not None:
            init.xavier_normal_(self.interpolation_func.weight.data)
            init.constant_(self.interpolation_func.bias.data, 0)     
    
    def forward(self, inputs, loss = None, last_edge_value=None):
        if last_edge_value is not None:
            output = F.tanh(self.edge_update_func(inputs))
            sigma = self.interpolation_func(inputs)
            output = F.sigmoid(sigma) * output + (1 - F.sigmoid(sigma)) * last_edge_value
        else:
            output = F.tanh(self.edge_update_func(inputs))
        return output

class Edge(nn.Module):
    def __init__(self, updater_set, hidden_size, edge_task_connect=None, edge_layer_connect=None, init_value=None):
        super(Edge, self).__init__()
        if init_value == None:
            self.edge_task_mask_value = torch.ones((1, hidden_size)).to(device)
            self.edge_layer_mask_value = torch.ones((1, hidden_size)).to(device)
        else:
            self.edge_task_mask_value = init_value[0]
            self.edge_layer_mask_value = init_value[1]
        
        self.loss_factor = lambda u: torch.exp(-torch.abs(1.414*3.1415926*u))

        if edge_task_connect is not None:
            self.edge_task_connnect = edge_task_connect
        elif edge_layer_connect is not None:
            self.edge_layer_connnect = edge_layer_connect
        
        for i, updater in enumerate(updater_set):
            self.add_module('layer{}'.format(i), updater)
        self.num_layer = i + 1 
    
    def update(self, mode, *inputs):
        loss, h1, h2 = inputs
        self.edge_task_mask_value = self.loss_factor(loss) * self.edge_task_mask_value 
        inputs = torch.cat([h1,h2],1)
        # print (inputs.size())
        # pdb.set_trace()
        if mode == 'task':
            temp_out = self._modules['layer{}'.format(0)](inputs)
            for i in xrange(1, self.num_layer - 1):
                temp_out = self._modules['layer{}'.format(i)](temp_out) 
            self.edge_task_mask_value = self._modules['layer{}'.format(i+1)](temp_out, self.edge_task_mask_value)
        elif mode == 'layer':
            temp_out = self._modules['layer{}'.format(0)](inputs)
            for i in xrange(1, self.num_layer - 1):
                temp_out = self._modules['layer{}'.format(i)](temp_out) 
            self.edge_task_mask_value = self._modules['layer{}'.format(i+1)](temp_out, self.edge_task_mask_value)

class Graph(nn.Module):
    def __init__(self, num_task, num_layer, num_updater_layer, grad_size_dict, num_kernel, hidden_size, edge_middle_size, node_middle_size):
        super(Graph, self).__init__()
        self.num_layer = num_layer
        self.num_task = num_task
        self.node_dict = defaultdict(list)
        self.task_edge_dict = defaultdict(list)
        self.layer_edge_dict = defaultdict(list)

        self.share_net = torch.nn.Sequential()

        num_node_updater_layer, num_node_output_layer, num_task_edge_updater_layer, num_layer_edge_updater_layer = num_updater_layer
        node_updater_list = []
        task_edge_updater_list = []
        layer_edge_updater_list = []

        grad_size = []
        for value in grad_size_dict.values():
            if not value[0] in grad_size:
                grad_size.append(value[0])

        for gs in grad_size:
            one_layer_updater = []
            lstm_updater_list = []
            for i in xrange(num_node_updater_layer):
                updater = NodeUpdater(2*gs, hidden_size).to(device)
                self.share_net.add_module("submodel_node_update{}".format(i), updater)
                lstm_updater_list.append(updater)
            one_layer_updater.append(lstm_updater_list)
            output_list = []
            for i in xrange(num_node_output_layer):
                if i != num_node_output_layer -1: 
                    outsize = node_middle_size[i]
                else:
                    outsize = gs
                if i == 0:
                    insize = hidden_size
                else:
                    insize = node_middle_size[i-1]
                outputlayer = GenerateGradientLayer(insize, outsize).to(device)
                self.share_net.add_module("submodel_node_output{}".format(i), outputlayer)
                output_list.append(outputlayer)
            one_layer_updater.append(output_list)
            
            node_updater_list.append(one_layer_updater)


        def get_edge_updater_list(num_layer, hidden_size, middle_size):
            edge_updater_list = []
            if middle_size:
                for i in xrange(num_layer - 1):
                    if i == 0:
                        in_size = 2 * hidden_size
                        out_size = edge_middle_size[i]

                    else:
                        in_size = out_size
                        out_size = edge_middle_size[i]
                    
                    updater = EdgeUpdater(in_size, middle_size = out_size).to(device)
                    edge_updater_list.append(updater)

                in_size = out_size
                out_size = hidden_size
                updater = EdgeUpdater(in_size, output_size = out_size).to(device)
                edge_updater_list.append(updater)
            else:
                updater = EdgeUpdater(2 * hidden_size, output_size = hidden_size).to(device)
                edge_updater_list.append(updater)
            return edge_updater_list
        

        task_edge_updater_list = get_edge_updater_list(num_task_edge_updater_layer, hidden_size, edge_middle_size)
        for i, updater in enumerate(task_edge_updater_list):
            self.share_net.add_module("submodel_task_edge{}".format(i), updater)
        layer_edge_updater_list = get_edge_updater_list(num_layer_edge_updater_layer, hidden_size, edge_middle_size)
        for i, updater in enumerate(layer_edge_updater_list):
            self.share_net.add_module("submodel_layer_edge{}".format(i), updater)
        
        for i in xrange(self.num_task):
            q = -1
            for j in xrange(self.num_layer):
                if j == 0 or grad_size_dict[j] != grad_size_dict[j-1]:
                    q += 1
                node = Node((i,j), node_updater_list[q], hidden_size)
                self.node_dict[(i,j)].append(node)
                for k in xrange(j+1):
                    edge = Edge(layer_edge_updater_list, hidden_size, edge_layer_connect=((i,j), (i,k)))
                    self.layer_edge_dict[((i,j), (i,k))].append(edge)
                for m in xrange(i+1):
                    edge = Edge(task_edge_updater_list, hidden_size, edge_task_connect=((i,j),(m,j))) 
                    self.task_edge_dict[((i,j), (m,j))].append(edge)

    def update_edge(self, mode, loss):
        if mode == 'task_level':
            for j in xrange(self.num_layer):
                for i in xrange(self.num_task):
                    for k in xrange(i):
                        edge_index = ((i,j), (k,j))
                        self.task_edge_dict[edge_index][0].update(mode, loss[i], self.node_dict[(i,j)][0].h, self.node_dict[(k,j)][0].h)

        elif mode == 'layer_level':
            for i in xrange(self.num_task):
                for j in xrange(self.num_layer):
                    for k in xrange(j):
                        edge_index = ((i,j), (i,k))
                        self.layer_edge_dict[edge_index][0].update(mode, loss[i], self.node_dict[(i,j)][0].h, self.node_dict[(i,k)][0].h)

    def update_node(self, grad_grid):
        for i in xrange(self.num_task):
            for j in xrange(self.num_layer):
                # pdb.set_trace()
                self.node_dict[(i,j)][0].update(grad_grid[(i,j)][0])

    #no explicit gradient fusing
    def propagate(self, mode):
        if mode == 'task_level':
            task_hidden_grid = defaultdict(list)
            for j in xrange(self.num_layer):
                for i in xrange(self.num_task):
                    for k in xrange(i+1):
                        temp_h_k = self.task_edge_dict[((i,j), (k,j))][0].edge_task_mask_value * self.node_dict[(i,j)][0].h
                        if len(task_hidden_grid[(k,j)]) != 0: task_hidden_grid[(k,j)][0] += temp_h_k
                        else: task_hidden_grid[(k,j)].append(temp_h_k)
                        if i != k:
                            temp_h_i = self.task_edge_dict[((i,j), (k,j))][0].edge_task_mask_value * self.node_dict[(k,j)][0].h
                            if len(task_hidden_grid[(i,j)]) != 0: task_hidden_grid[(i,j)][0] += temp_h_i
                            else: task_hidden_grid[(i,j)].append(temp_h_i)
                
                for m in xrange(self.num_task):
                    task_hidden_grid[(m,j)][0] /= self.num_task

            return task_hidden_grid

        elif mode == "layer_level":
            layer_hidden_grid = defaultdict(list)
            for i in xrange(self.num_task):
                for j in xrange(self.num_layer):
                    for k in xrange(j+1):
                        temp_h_k = self.layer_edge_dict[((i,j), (i,k))][0].edge_layer_mask_value * self.node_dict[(i,j)][0].h
                        if len(layer_hidden_grid[(i,k)]) != 0: layer_hidden_grid[(i,k)][0] += temp_h_k
                        else: layer_hidden_grid[(i,k)].append(temp_h_k)
                        if j != k:
                            temp_h_i = self.layer_edge_dict[((i,j), (i,k))][0].edge_layer_mask_value * self.node_dict[(i,k)][0].h
                            if len(layer_hidden_grid[(i,j)]) != 0: layer_hidden_grid[(i,j)][0] += temp_h_i
                            else: layer_hidden_grid[(i,j)].append(temp_h_i)
        

                for m in xrange(self.num_layer):
                    layer_hidden_grid[(i,m)][0] /= self.num_layer
            return layer_hidden_grid

        else: raise NotImplementedError

    def update_hidden_state(self, hidden_state_grid):
        for i in xrange(self.num_task):
            for j in xrange(self.num_layer):
                self.node_dict[(i,j)][0].h = hidden_state_grid[(i,j)][0]   

    def parameters(self):
        param_list = []

        for i in xrange(self.num_task):
            for j in xrange(self.num_layer):
                node = self.node_dict[(i,j)][0]
                param_list.append(node.parameters())
                for k in xrange(j+1):
                    edge = self.layer_edge_dict[((i,j), (i,k))]
                    param_list.append(edge.parameters())
                for m in xrange(i+1):
                    edge = self.task_edge_dict[((i,j), (m,j))]
                    param_list.append(edge.parameters())
        
        return param_list

class GraphIO(object):
    def __init__(self, model, num_task, hidden_size):
        self.model = model
        self.num_task = num_task
        self.grad_grid = defaultdict(list)
        
    def get_grad_from_model(self):
        self.grad_grid.clear()
        for i in xrange(self.num_task):
            for j, layer in enumerate(self.model[i].children()):
                weight_grad = preprocess(layer.weight.grad.detach().view(-1).unsqueeze(0))
                if layer.bias is not None:
                    bias_grad = preprocess(layer.bias.grad.detach().view(-1).unsqueeze(0))
                    whole_grad = torch.cat([weight_grad, bias_grad], 1)
                else:
                    whole_grad = weight_grad
                self.grad_grid[(i,j)].append(whole_grad)

    def update_grad_grid(self, node):
        index = node.node_id
        new_grad = node.output_grad()
        self.grad_grid[index][0] = new_grad

    def set_grad_to_model(self):
        for i in xrange(self.num_task):
            for j, layer in enumerate(self.model[i].children()):
                wsize = layer.weight.grad.data.view(-1).size()[0]
                wgrad = self.grad_grid[(i,j)][0][0][:wsize].unsqueeze(0)
                layer.weight.grad.data.copy_(wgrad.view_as(layer.weight.grad.data))
                gc.collect()
                if layer.bias is not None:
                    bsize = layer.bias.grad.data.view(-1).size()[0]
                    bgrad = self.grad_grid[(i,j)][0][0][-bsize:].unsqueeze(0)
                    layer.bias.grad.data.copy_(bgrad.view_as(layer.bias.grad.data))
                    gc.collect()




class GraphNetworkOnGradient(nn.Module):
    def __init__(self, post_model_set, num_task, num_layer, num_updater_layer, num_kernel,
                hidden_size, edge_middle_size, node_middle_size):
        super(GraphNetworkOnGradient, self).__init__()
        self.num_task = num_task
        self.num_layer = num_layer
        
        grad_size_dict = self._get_grad_size(post_model_set[0])

        self.grad_processor = GraphIO(post_model_set, num_task, hidden_size)
        self.gn = Graph(num_task, num_layer, num_updater_layer, grad_size_dict, num_kernel, hidden_size, edge_middle_size, node_middle_size)

    def forward(self, loss):
        self.grad_processor.get_grad_from_model()

        self.gn.update_edge("task_level", loss)
        self.gn.update_edge("layer_level", loss)
        
        self.gn.update_node(self.grad_processor.grad_grid)
        
        task_hidden_grid = self.gn.propagate("task_level")
        self.gn.update_hidden_state(task_hidden_grid)
        layer_hidden_grid = self.gn.propagate("layer_level")
        self.gn.update_hidden_state(layer_hidden_grid)
    
        for node in self.gn.node_dict.values():
            self.grad_processor.update_grad_grid(node[0])
        
        self.grad_processor.set_grad_to_model()

    def _get_grad_size(self, model):
        grad_size_dict = defaultdict(list)
        for j, layer in enumerate(model.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                gsw = layer.weight.data.view(-1).size()[0]
                grad_size_dict[j].append(gsw)
                try:
                    gsb = layer.bias.data.view(-1).size()[0]
                except AttributeError:
                    continue
                else:
                    grad_size_dict[j][0] += gsb
        
        return grad_size_dict

class SGD(object):
    def __init__(self, learning_rate):
        super(SGD, self).__init__()
        self._learning_rate = learning_rate

    def __call__(self, layer):
        with torch.no_grad():
            new = layer.weight.data - self._learning_rate * layer.weight.grad.data
        return layer.weight.data.copy_(new)

class Estimator(object):
    def __init__(self, learning_rate, input_channel_size, 
                N_way, N_shot, N_test,
                num_updater_layer, num_kernel,
                hidden_size, 
                edge_middle_size, node_middle_size, 
                mean, std, save_root, use_cuda):
        super(Estimator, self).__init__()
        
        if use_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        self.meta_model = BaseModel(input_channel_size, N_way)
        self.meta_model.to(device)
        
        self.num_train_task = N_shot * N_way
        self.num_test_task = N_test
        self.num_layer = self._get_num_layer()
        
        self.model_list = []
        for _ in xrange(self.num_train_task + self.num_test_task):
            model = BaseModel(input_channel_size, N_way)
            model.to(device)
            self.model_list.append(model)
    
        self.gnog = GraphNetworkOnGradient(self.model_list, self.num_train_task + self.num_test_task, 
                self.num_layer, num_updater_layer, num_kernel, hidden_size, edge_middle_size, node_middle_size)
        self.gnog.to(device)
        
        self.sgd_optimizer = SGD(learning_rate[0])

        self.gnog_net = self.gnog.gn.share_net
        self.optimizer_for_gnog = optim.Adam(self.gnog_net.parameters(), lr = learning_rate[1])
        self.optimizer_for_meta_model = optim.Adam(self.meta_model.parameters(), lr = learning_rate[2])

        self.mean = mean
        self.std = std

        self.best_acc = 0
        self.save_root = save_root

        self.device = device
    
    def train_(self, train_task, test_task):
        self.meta_model.train()
        for model in self.model_list:
            model.train()
        self._clear_hc()
        self._update_model_through_gnog(train_task)        
        self._update_gnog_params(train_task)
        self._update_meta_model(test_task)

    def test_(self, train_task):
        self.meta_model.train()
        for model in self.model_list:
            model.train()
        self._clear_hc()
        self._update_model_through_gnog(train_task)
    
    def evaluate(self, test_task):
        total_loss = 0.
        num_correct = 0.
        with torch.no_grad():
            for i, (task, label) in enumerate(test_task):
                self.model_list[self.num_train_task + i].eval()
                label_p = self.model_list[self.num_train_task + i](task)
                loss = F.nll_loss(label_p, label)
                total_loss += loss.item()
                predict = torch.argmax(label_p, dim=1)
                num_correct += (predict == label).sum().item()
        
        acc = float(num_correct) / (task.size()[0] * self.num_test_task)
        # loss = float(total_loss) / (task.size()[0] * self.num_test_task)
        return acc, total_loss

    def display(self, phase, epoch, batch, acc, loss):
        print ("{}: Epoch:{} Batch:{} Acc:{} Loss:{}".format(phase, epoch, batch, acc, loss))
    
    def save_model_params(self, acc):
        if acc < self.best_acc:
            self.best_acc = acc
            gnog_save_path = os.path.join(self.save_root, "gnog")
            meta_model_save_path = os.path.join(self.save_root, "meta_model")
            torch.save(self.gnog_net.state_dict(), gnog_save_path)
            torch.save(self.meta_model.state_dict(), meta_model_save_path)

    def _compute_taskwise_grad(self, train_task):
        loss_of_every_task = []
        for i, (task, label) in enumerate( train_task ): 
            #reset the task-based model list according to the new meta model
            self._copy_params_to(self.model_list[i])
            label_p  = self.model_list[i](task)
            loss = F.nll_loss(label_p, label)
            loss_of_every_task.append(loss.item()) 
            
            loss.backward()

        loss_mean = reduce(lambda x,y: x+y, loss_of_every_task)/len(loss_of_every_task)

        for i in xrange(self.num_test_task):
            self._forge_grad_to(self.model_list[self.num_train_task + i])
            loss_of_every_task.append(loss_mean)
        
        loss_of_every_task = map(lambda u: torch.tensor(u).to(device), loss_of_every_task)

        return loss_of_every_task

    def _update_model_through_gnog(self, train_task):
        self.gnog(self._compute_taskwise_grad(train_task))
        for model in self.model_list:
            for layer in model.children():
                self.sgd_optimizer(layer)
    
    def _update_gnog_params(self, train_task):
        total_loss = 0
        for i, (task, label) in enumerate( train_task ): 
            label_p = self.model_list[i](task)
            loss = F.nll_loss(label_p, label)
            loss.backward()
        self.optimizer_for_gnog.zero_grad()
        self.optimizer_for_gnog.step()
    
    def _update_meta_model(self, test_task):
        for i, (task, label) in enumerate( test_task ):
            label_p = self.model_list[self.num_train_task + i](task)
            loss = F.nll_loss(label_p, label)
            loss.backward()
        self.optimizer_for_meta_model.zero_grad()
        self.optimizer_for_meta_model.step()

    def _clear_hc(self):
        for i in xrange(self.num_train_task + self.num_test_task):
            for j in xrange(self.num_layer):
                self.gnog.gn.node_dict[(i, j)][0].h.fill_(0)
                self.gnog.gn.node_dict[(i, j)][0].c.fill_(0)
    
    def _forge_grad_to(self, model):
        for layer in model.children():
            fake_grad = (torch.randn(layer.weight.data.size()) * self.std + self.mean).to(self.device)
            layer.weight.grad = fake_grad
            if layer.bias is not None:
                fake_grad_b = (torch.randn(layer.bias.data.size()) * self.std + self.mean).to(self.device)
                layer.bias.grad = fake_grad_b

    def _copy_params_to(self, model):
        with torch.no_grad():
            for fl, tl in zip(self.meta_model.parameters(), model.parameters()):
                tl.data.copy_(fl.data)
    
    def _get_num_layer(self):
        n = 0
        for _ in self.meta_model.children():
            n += 1
        print ("num_layer:{}".format(n))
        return n