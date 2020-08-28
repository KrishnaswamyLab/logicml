
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace
from graphviz import Source
import pydot
from collections import OrderedDict
import numpy as np
from logicml.code.pipeline.binarized_modules import  *
from logicml.code.pipeline.utils import *
from logicml.code.pipeline.handler import *


class LogicNN(nn.Module):
    def __init__(self, args, num_features=9, num_classes=2, hidden_layer_outnodes=None, layer_connections_dict=None, in_features_dict=None):
        '''
            Initializing neural network for intersection with logic. 
            Can either be fully connected for initialization (when layer_connections_dict=None) or include more layer structures and skip-connections when created from logic.
            NOTE: we encode the input and hidden layers differently than the output layer, i.e. the hidden layers and input layer will be created based on layer_out_nodes and the output layer, based on num_classes
            Args: 
                num_features: number of input attributes (columns), depending on data set (e.g. 9 for coimbra dataset)
                num_classes: the number of classes to be classified, depending on data set (e.g. 2 for coimbra dataset)
                hidden_layer_outnodes: a list of nodes per layer (coming from the logic creation) - not the final layer; if None: args.hidden_layer_outnodes will be taken (e.g. for network initialization)
                layer_connections_dict: a dictionary with key = number of layer (all except for 0) and value = list of incoming layer features to also model skip connections, if None: just feed forward architecture (take output from previous layer)
                in_features_dict: a dictionary with key = int number of layer (all except for 0) and value = number of incoming features to also model skip connections, if None: just feed forward architecture (take output from previous layer)
        '''
        super(LogicNN, self).__init__()

        # Argparser Arguments
        self.args = args
        self.num_classes = num_classes
        self.num_features = num_features

        if hidden_layer_outnodes: # when explicitly given from the outside, e.g. from logic (net with skip-connections) or snapshot
            self.hidden_out_nodes = hidden_layer_outnodes
        else: # initialization as fully-connected architecture
            self.hidden_out_nodes = self.args.hidden_layer_outnodes
        
        self.hidden_num_layers = len(self.hidden_out_nodes) 

        # depending on provided information we have a fully-connected or a skip-layer architecture for architecture learning
        if layer_connections_dict and in_features_dict: 
            self.skip_connections = True
            self.layer_connections_dict = layer_connections_dict 
            self.in_features_dict = in_features_dict 
        else: 
            self.skip_connections = False 
            self.layer_connections_dict = {} # will be filled in the following
            self.in_features_dict = {} # will be filled in the following

        self.layers = nn.ModuleList() 

        # define the network output names and input names for visualization that will be used within the trainer
        self.input_names = ['input']
        self.output_names = []


        for i in range(self.hidden_num_layers): # handling all layers except for the last one 
            # list to create a sequential module
            sequential_list = []

            # handle fully connected layer
            if i == 0: # first layer is different
                if self.args.binary_weights: 
                    fc = BinarizeLinear(num_features, self.hidden_out_nodes[0], num_features=num_features, stochastic_bin=self.args.stochastic_binarization ,bias=self.args.bias)
                else: 
                    fc = nn.Linear(num_features, self.hidden_out_nodes[0], bias=self.args.bias)
            else: 
                if self.skip_connections: # with modeling of skip connections
                    if self.args.binary_weights: 
                        fc = BinarizeLinear(self.in_features_dict[i], self.hidden_out_nodes[i], num_features=num_features, stochastic_bin=self.args.stochastic_binarization, bias=self.args.bias)
                    else: 
                        fc = nn.Linear(self.in_features_dict[i], self.hidden_out_nodes[i], bias=self.args.bias)
                else: # just feed forward architecture
                    if self.args.binary_weights: 
                        fc = BinarizeLinear(self.hidden_out_nodes[i-1], self.hidden_out_nodes[i], num_features=num_features, stochastic_bin=self.args.stochastic_binarization, bias=self.args.bias)
                    else: 
                        fc = nn.Linear(self.hidden_out_nodes[i-1], self.hidden_out_nodes[i], bias=self.args.bias)
                    self.layer_connections_dict[i] = [i-1]
                    self.in_features_dict[i] = self.hidden_out_nodes[i-1]

            nn.init.xavier_uniform_(fc.weight)
            if self.args.bias: 
                nn.init.zeros_(fc.bias)
            sequential_list.append(('fc{}'.format(i), fc))

            # handle activation layer
            if str.lower(self.args.hidden_activation) == 'htanh': 
                sequential_list.append(('htanh{}'.format(i), nn.Hardtanh())) # the htanh-activations are the ones that need to be collected for the logic creation within architecture learning
                self.output_names.append('htanh{}'.format(i)) # needed for visualization in visualize_net()
            else: 
                sequential_list.append(('relu{}'.format(i), nn.ReLU())) # the relu-activations are the ones that need to be collected for the logic creation within random forest, LogicNet or the NN translation
                self.output_names.append('relu{}'.format(i)) # needed for visualization in visualize_net()
            
            # treat fc and activation as one stage
            sequential_module = nn.Sequential(OrderedDict(sequential_list))
            self.layers.add_module('FC_Stage{}'.format(i), sequential_module)

            # handle dropout layer and treat it as one module
            if self.args.dropout and i == self.hidden_num_layers-1: # Dropout before last final layer
                sequential_module = nn.Sequential(OrderedDict([('dropout{}'.format(i), nn.Dropout(0.5))]))
                self.layers.add_module('Dropout_Stage{}'.format(i), sequential_module)
            
            # handle batch norm layer
            if self.args.batchnorm: 
                sequential_module = nn.Sequential(OrderedDict([('batchnorm{}'.format(i), nn.BatchNorm1d(self.hidden_out_nodes[i]))]))
                self.layers.add_module('Batchnorm_Stage{}'.format(i), sequential_module)

        # treat final block: 
        sequential_list = []
        # add output name for final layer
        self.output_names.append('output')

        # start handling final layer
        if self.num_classes == 2 and self.args.sigmoid:  # last layer with only one node and sigmoid only in binary classfication case
            self.out_nodes_number = 1
        else: 
            self.out_nodes_number = self.num_classes

        # creating fc block
        # NOTE: on last layer we don't use BinarizeLinear layer but nn.Linear because of softmax / sigmoid
        if self.skip_connections: # with skip connections
            sequential_list.append(('fc_final', nn.Linear(self.in_features_dict[self.hidden_num_layers], self.out_nodes_number, bias=self.args.bias)))
        else: # just fully connected architecture
            sequential_list.append(('fc_final', nn.Linear(self.hidden_out_nodes[-1], self.out_nodes_number, bias=self.args.bias)))
            # encoding last layer also in dicts because this is not the case for initialized fc-layers
            self.layer_connections_dict[self.hidden_num_layers] = [self.hidden_num_layers-1]
            self.in_features_dict[self.hidden_num_layers] = self.hidden_out_nodes[-1]

        # sigmoid or softmax block
        if self.out_nodes_number == 1: 
            sequential_list.append(('sigmoid', nn.Sigmoid()))
        else: 
            sequential_list.append(('softmax', nn.Softmax(dim=-1)))

        sequential_module = nn.Sequential(OrderedDict(sequential_list))

        self.layers.add_module('Final_Stage{}'.format(self.hidden_num_layers), sequential_module)

        # Default loss definiton (used in run.py)
        if self.out_nodes_number == 1:  # last layer with only one node and sigmoid only in binary classfication case for architecture learning
            self.default_loss = nn.BCELoss() # use with sigmoid layer before hand 
        else: 
            self.default_loss = nn.CrossEntropyLoss() # use with softmax layer before hand 


    def forward(self, x):

        # TODO: maybe introduce additional return of a list with layer_numbers that correspond to the same positions of activations to return, once we have more complex architectures or more layer types
        # because currently assumed that the index of a tensor in the list activations_to_return is also the layer_number

        activations = {} # key = layer_number and value = torch tensor that forms the activations - # used for computation, was especially needed to handle skip-connections
        activations_to_return = [] # might differ from activations dict above because only fc layers and output layer should be returned (above can contain batchnorm and dropout as well) 
        x = x.view(-1, self.num_features)
        out = None

        for layer_module in self.layers:
            if 'fc' in str(layer_module): 
                split_key = 'fc'
            elif 'dropout' in str(layer_module):
                split_key = 'dropout'
            else: 
                split_key = 'batchnorm'
            
            if 'final' in str(layer_module): 
                layer_number = self.hidden_num_layers
            else: 
                layer_number = str(layer_module).split(split_key)[-1].split(')')[0]
                layer_number = int(layer_number)

            if 'dropout' in str(layer_module) or 'batchnorm' in str(layer_module): # all batchnorm or dropout layers, take ativations from fc layer with same layer number and then overwrite the activation
                x = activations[layer_number]

            elif layer_number != 0: # all fully connected layers, except the first one need to get the right previous activations to be processed
                # first layer can only receive batch input as declared in x in beginning of forward method
                # for all other layers concatenate previous layers outputs according to self.layer_connections_dict (if we have skip connections)
                connections = sorted(self.layer_connections_dict[layer_number])
                x = activations[connections[0]] # first element in connection
                if len(connections) > 1: # concatenate all other previous activations to form the input of the layer if we have skip connections
                    # concatenate higher layer index beneath lower index, i.e. ascending (has advantage that we then really now which weight row/column refers to what)
                    for i in range(1, len(connections)): 
                        x = torch.cat((x, activations[connections[i]]), dim=1)
            
            # update the activations dict for every layer type (no matter if batchnorm, dropout, fc or first or other layer)
            activations[layer_number] = layer_module(x)

            # update activations to return dict only with activations from fc layers that are not the last layer
            if 'fc' in str(layer_module) and not ('final' in str(layer_module)): 
                # activations_to_return.append(activations[layer_number].clone().detach())
                activations_to_return.append(activations[layer_number])

        out = activations[self.hidden_num_layers]
        return activations_to_return, out # returning a sequence of a list of activations from hidden fc layers (index in list=layer_number) and out=activation from final layer (torch.tensor)


    def training_routine(self, epoch, batch, optimizer, criterion, return_logic_infos=False):

        data, target = batch['features'], batch['labels']

        if isinstance(criterion, nn.BCELoss): # means a two-class classification problem with one output node and sigmoid
            data, target = torch.from_numpy(data).float(), torch.from_numpy(target).float()
        else: # means multiple output nodes and a softmax with CrossEntropy Function 
            data, target = torch.from_numpy(data).float(), torch.from_numpy(target).long().squeeze()

        if self.args.cuda:
            data, target = data.cuda(), target.cuda()
        
        data, target = Variable(data), Variable(target)

        # returning three activations and ouput of the model for given data_batch
        # output = torch tensor and activations = dictionary
        activations, output = self(data)

        # do the processing for the logic stuff and dont do any optimization, just collect the activations

        if return_logic_infos and (self.args.logic_net or self.args.nn_translation or (self.args.random_forest and self.args.rf_bitwise_training)): 
            # NOTE: this will already return the activations and outputs as numpy arrays with quantized binary string representations, according to the given quantization scheme 
            accumulated_activations = process_torch_tensors_to_numpy_lists_bin_str_repr(activations, self.args.cuda, self.args.fractional_bits, self.args.total_bits, activations_output_mode=0)
            accumulated_outputs = process_torch_tensors_to_numpy_lists_bin_str_repr([output], self.args.cuda, self.args.fractional_bits, self.args.total_bits, activations_output_mode=1) # NOTE: need to pack output in a list [output]

        elif return_logic_infos and self.args.random_forest and not self.args.rf_bitwise_training: 
            # the case where we train random forests on the activations, but not in a bitwise manner
            # in this case the activations and outputs need to be floats and not string representations
            accumulated_activations = process_torch_tensors_to_numpy_lists_bin_int_repr(activations, self.args.cuda, self.args.fractional_bits, self.args.total_bits, activations_output_mode=0)
            accumulated_outputs = process_torch_tensors_to_numpy_lists_bin_int_repr([output], self.args.cuda, self.args.fractional_bits, self.args.total_bits, activations_output_mode=1) # NOTE: need to pack output in a list [output]


        else: # dont do logic stuff and do normal training steps
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()

            # the following will only has impact when the network was initialized with a BinarizeLinear layer (hence, is a binarized neural network)
            # weights copied for binarization, but optimizer step done on "continous" latent weights
            for p in list(self.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org.data)

            # perform optimizer step
            optimizer.step()

            # the following will only has impact when the network was initialized with a BinarizeLinear layer (hence, is a binarized neural network)
            # binarization of the updated latent weights to use them again in the forward pass
            for p in list(self.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        if return_logic_infos: 
            return {
                'activation_features' : accumulated_activations, 
                'outputs' : accumulated_outputs}

        else: 
            return {'loss' : loss.item()}

    def testing_routine(self, batch, criterion):

        data, target = batch['features'], batch['labels']

        if isinstance(criterion, nn.BCELoss): # means a two-class classification problem with one output node and sigmoid
            data, target = torch.from_numpy(data).float(), torch.from_numpy(target).float()
        else: # means multiple output nodes and a softmax with CrossEntropy Function 
            data, target = torch.from_numpy(data).float(), torch.from_numpy(target).long().squeeze()

        if self.args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            data, target = Variable(data), Variable(target)

        # forward pass of inputs and evaluation
        # NOTE: due to returns from forward pass, we still have to take the activations, although we don't use them here

        activations, output = self(data)
        test_loss = criterion(output, target).item() # sum up batch loss

        if isinstance(criterion, nn.BCELoss): # means a two-class classification problem with one output node and sigmoid
            pred = output >= 0.5
            pred = pred.byte()
        else: # means multiple output nodes and a softmax with CrossEntropy Function 
            _, pred = output.max(-1) # returns max, argmax

        correct = torch.sum(pred == target)

        return {'loss' : test_loss, 'correct': correct, 'predictions' : pred.cpu().numpy().tolist(), 'labels' : target.cpu().numpy().tolist()}