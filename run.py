import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
from logicml import *
import torchvision
import sys


if __name__ == '__main__':

    # ---------------- Argparser Handling ----------------
    used_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Combining Machine Learning With Logic Synthesis - Training')
    args_handler = ArgsHandler(parser, used_args)
    parser = args_handler.get_parser()
    args = args_handler.get_args()

    for i in range(args.num_runs): # run several times
        # adapt experiment name if more than one run
        if args.num_runs > 1: 
            args.experiment_name = args.original_experiment_name + '_run{}'.format(i)

        # ---------------- Handler  ----------------
        handler = Handler(name=args.experiment_name, verbosity=args.verbosity, basepath=args.basepath, overwrite=args.handler_overwrite)

        # ---------------- Data Reader ----------------
        readers = {
            'coimbra': CoimbraReader,
            'mnist' : MNISTReader}

        reader = readers[str.lower(args.datareader)](args)
        reader.shuffle_and_tie_new_batches()

        # ---------------- Network Definitions ----------------
        nets = {
            'fcnn': LogicNN} # initialization as fully connected network to be used with architecture learning option
        
        net = nets[str.lower(args.architecture)](args, num_features=reader.num_features, num_classes=reader.num_classes)  

        # ---------------- Optimizer Definition ----------------             
        optimizers = {
                'adam' : optim.Adam,
                'sgd' : optim.SGD}   

        if str.lower(args.optimizer) == 'sgd': # including momentum term
            optimizer = optimizers[str.lower(args.optimizer)](net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else: 
            optimizer = optimizers[str.lower(args.optimizer)](net.parameters(), lr=args.lr)

        # ---------------- Loss Definition ----------------
        criteria = {
                'bce' : nn.BCELoss(),
                'default' : net.default_loss}

        criterion = criteria[str.lower(args.loss)]

        # ---------------- Logic Processor ----------------
        logic_processor = LogicProcessing(handler, args=args)

        # ---------------- Training ----------------
        trainer = Trainer(handler, args, reader, logic_processor, net, optimizer, criterion)
        trainer.write_config_summary(str(args_handler))
        
        # Direct Random Forest routine
        if args.random_forest_direct: # handling direct random forest
            model = trainer.train_direct_rf()
            if not args.just_logic: 
                trainer.test_direct_rf(model) 

        # All routines based on neural network
        else: # handling neural network and possible logic derivations
            if not args.just_test:
                trainer.train_nn() # including validation, if this is declared in the argparser arguments
            if not args.just_logic: 
                trainer.test() # normal testing procedure of the neural network
        if args.random_forest or args.logic_net or args.nn_translation or args.random_forest_direct:
            trainer.test_logic() # testing procedure of the logic