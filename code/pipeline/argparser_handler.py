import argparse
import torch

class ArgsHandler: 

    def __init__(self, parser, used_args): 
        '''
            This class handles everything that has to do with the args provided by the ArgParser. 
            It includes defining the arguments, assertions, default settings and configuartion summaries. 
            Args: 
                parser: the Python ArgumentParser-class object 
                used_args: the used arguments string that is collected with sys.argv[1:] in the Python script that is actually called
            Returns: 
                sequence of the changed parser and args-dict

        '''
        assert isinstance(parser, argparse.ArgumentParser)
        self.parser = parser
        self.used_args = used_args # comes from the outside, from run.py script -->  sys.argv[1:]
        self.args = self.create_args() # creating the args and doing assertions
        self.args = self.default_handling(self.args) # reset some arguments and default based on the args and parameters that were used in command line

    def create_args(self):
        # ---------------- Argparser Settings ----------------
        # General Settings
        self.parser.add_argument('--experiment_name', type=str, default='default_experiment', help='The name of this experiment - will create folder for results')
        self.parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
        self.parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
        self.parser.add_argument('--log_interval', type=int, default=1, help='How many batches to wait before logging training status')
        self.parser.add_argument('--verbosity', type=str, default='info', help='Sets the verbosity level for logging - can be one of the following: [debug, info]')
        self.parser.add_argument('--basepath', type=str, default=None, help='Basepath for handler to store results and logs (absolute path including last slash); if None: results folder in this GIT-Repro will be used')
        self.parser.add_argument('--no_handler_overwrite', action='store_true', default=False, help='Overwriting the previous logs under same experiment name')
        self.parser.add_argument('--tensorboard_disable', '--tb_dis', action='store_true', help='Parameter to set, if no tensorboard logging should be done.')
        self.parser.add_argument('--num_runs', type=int, default=1, help='Random seed (default: 1)')
        self.parser.add_argument('--results_subfolder', type=str, default=None, help='A string for a subfolder in the results folder, to which the results csv will be written.')
        self.parser.add_argument('--nn_results_filename', '--nn_csv', type=str, default='nn_results', help='A string for the csv file-name to which the results of neural network testing will be written.')
        self.parser.add_argument('--direct_rf_results_filename', '--direct_rf_csv', '--rf_csv', type=str, default='direct_rf_results', help='A string for the csv file-name to which the results of direct random forest testing will be written.')
        self.parser.add_argument('--resource_monitoring', '--rm', action='store_true', help='Parameter to set, if resource management should actively be monitored.')
        self.parser.add_argument('--snapshotting', action='store_true', default=False, help='Enables snapshotting.')
        self.parser.add_argument('--snapshotting_overwrite', action='store_true', default=False, help='Can be called if an already existing snapshot that runs under the same experiment name should be overwritten.')
        self.parser.add_argument('--snapshotting_best', action='store_true', default=False, help='Can be called if besides latest, also the best model in terms of performance on validation data set should be snapshotted.')
        self.parser.add_argument('--snapshotting_interval', type=int, default=1, help='The epoch interval for snapshotting the latest and best models during training.')
        self.parser.add_argument('--snapshotting_criterion', type=str, default='min_loss', help='Defines the export criterion for snapshotting best net. Use following structure <max/min>_<sth in return_dict from net validation>, currently: loss or accuracy')
        self.parser.add_argument('--load_snapshot', type=str, default=None, help='In case you want to load a snapshot before training it further, hand over the path to the folder where the snapshotted files are located (with or without last slash). Be careful, it reconstructs the network and data reader setting and will ignore other settings made! None: in case you dont want to load a snapshot')
        self.parser.add_argument('--just_test', action='store_true', help='Parameter to call if the network should not be trained, but just tested (e.g. in combination with loading a pre-trained snapshot).')
        self.parser.add_argument('--netron_export', '--netron', action='store_true', help='Parameter to call if the neural network structure should be exported to be visualized with NETRON (for architecture learning in each round).')

        # Arguments related to Data
        self.parser.add_argument('--datareader', '--reader', type=str, default='coimbra', help='Which Reader to use - choose between following: coimbra, mnist')
        self.parser.add_argument('--mnist_use_args_test_splits', '--mnist_split', action='store_true', default=False, help='Parameter to call if the test_split parameter from the args should also be used for MNIST. Otherwise just regular test data set.')
        self.parser.add_argument('--batch_size', '--bs', type=int, default=4, help='Input batch size for training and testing')
        self.parser.add_argument('--test_split', '--ts', type=float, default=0.2, help='The percentage of data that should be used as test data')
        self.parser.add_argument('--validation_split', '--vs', type=float, default=0.2, help='The percentage of training data that should be used as validation data')
        self.parser.add_argument('--enforce_new_data_creation', action='store_true', default=False, help='Enforcing that data arrays are not loaded from numpy array files when stored, but being overwritten')
        self.parser.add_argument('--data_normalization_mode', '--norm', type=int, default=4, help='Mode of Data Normalization - Choose between: [0: no normalization, 1: Z-Score Normalization, 2: Min-Max-Scaling, 3: Max-Abs-Scaling, 4: Robust-Scaling, 5: Power-Transforming, 6: Quantile-Transforming, 7: Independent Normalization].')
        self.parser.add_argument('--no_balanced_test_data', '--balanced', action='store_true', help='Parameter to call if (in case of using the arg splits) the test data should not be perfectly balanced among the classes')
        self.parser.add_argument('--unbalanced_class_train_keep', '--unbalanced_keep', type=float, default=0.5, help='The percentage of the most unbalanced class that should be kept for training, in case of the datareader creating a perfectly balanced test set.')

        # Arguments related to Network
        self.parser.add_argument('--architecture', type=str, default='fcnn', help='Which Network To Use as Initialization. Choose between following keywords: fcnn')
        self.parser.add_argument('--hidden_layer_outnodes', '--nodes', type=int, nargs='*', default=[9, 9, 9], help='Number of nodes per hidden and input layer for fcbnn architecture (except for last layer) [length of list decides for number of layers]')
        self.parser.add_argument('--batchnorm', action='store_true', help='Parameter to call if batch normalization should be done.')
        self.parser.add_argument('--dropout', action='store_true', help='Parameter to call if external dropout should be applied.')
        self.parser.add_argument('--no_bias', action='store_true', default=False, help='Disables biases in the fully connected layers')
        self.parser.add_argument('--binary_weights', action='store_true', default=False, help='Enables usage of binary weights in the fully connected layers, i.e. uses BinarizeLinear layers instead of nn.Linear layers (but still independent of activation function)')
        self.parser.add_argument('--bnn', action='store_true', default=False, help='Turns it into a binarized neural network, i.e. using binarized weights and htanh activation function.')
        self.parser.add_argument('--stochastic_binarization', action='store_true', default=False, help='Enables usage of stochastic binarization in weight binarization forward pass, otherwise: deterministic binarization is used (=sign function).')
        self.parser.add_argument('--hidden_activation', type=str, default='htanh', help='Which activation function to choose for layers that are not the last one. Choose between following keywords: htanh, relu')
        self.parser.add_argument('--sigmoid', action='store_true', default=False, help='When this paraemter is set and a binary classification is used, then the last layer has just one node, sigmoid is used and binary cross entropy loss.')

        # Arguments related to Training
        self.parser.add_argument('--epochs', '--N', type=int, default=100, help='Number of epochs to train') # TODO: maybe adapt default depending on whether it is architecture learning or not - in case of architecture learning --> 399
        self.parser.add_argument('--loss', type=str, default='default', help='The Loss used for the Network. Choose between following keywords: hinge, hingesqrt, bce, default')
        self.parser.add_argument('--optimizer', '--optim', type=str, default='adam', help='The Loss used for the Network. Choose between following keywords: adam, sgd')
        self.parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
        self.parser.add_argument('--no_lr_scheduling', action='store_true', default=False, help='Disables Learning Rate Scheduling')
        self.parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--weight_decay', '--decay', type=float, default=0.0, help='SGD weight decay')
        self.parser.add_argument('--no_validation', action='store_true', default=False, help='Disables validation')
        
        # Arguments related to Logic in general
        self.parser.add_argument('--just_logic', action='store_true', help='Parameter to call if the network should not be trained, but just logic should be created (e.g. in combination with loading a pre-trained snapshot).')
        self.parser.add_argument('--nn_translation', '--nnt', '--nn_trans', action='store_true', default=False, help='Parameter to call if the routine for training a NN, translating it to logic and simulating it should be executed.')
        self.parser.add_argument('--random_forest', '--rf', action='store_true', default=False, help='Parameter to call if the routine for training a NN, training random forest on activations, translating it to logic and simulating it should be executed.')
        self.parser.add_argument('--random_forest_direct', '--rf_direct', '--direct_rf', action='store_true', default=False, help='Parameter to call if a random forest should be trained directly on the data and then be translated to logic (without involving neural networks).')
        self.parser.add_argument('--logic_net', '--lgn', action='store_true', default=False, help='Parameter to call if the routine for training a NN, training LogicNet on activations, translating it to logic and simulating it should be executed.')
        self.parser.add_argument('--full_comparison', '--full', '--full_logic', action='store_true', default=False, help='Parameter to call if all of the following args should be set to True at the same time: random_forest, logic_net, nn_translation.')
        self.parser.add_argument('--aig_optimization', '--aig_optim', action='store_true', default=False, help='Parameter to call if AIG statistics should be used after optimizing the AIG, otherwise statistics of original AIG will be used.')
        self.parser.add_argument('--aig_optimization_command', '--aig_optim_cmd', type=str, default='syn2', help='The command from ABC with which to opimtimize the AIG. Choose between following keywords: syn2, dc2, mfs, mfs_advanced')
        self.parser.add_argument('--aig_export', action='store_true', default=False, help='Parameter to call if final AIG file should be stored.')
        self.parser.add_argument('--abc_verilog_export', action='store_true', default=False, help='Parameter to call if ABC should export the processed Verilog file.')
        self.parser.add_argument('--logic_results_filename', '--logic_csv', type=str, default='logic_results', help='A string for the csv file-name to which the results of logic testing will be written.')
        self.parser.add_argument('--total_bits', '--tb', type=int, default=8, help='The number of total bits to which the activations, weights and biases should be quantized.')
        self.parser.add_argument('--fractional_bits', '--fb', type=int, default=6, help='The number of bits from the total bits that should be used as fractional bits to which the activations, weights and biases should be quantized.')
        # NOTE: assumption: n = total bits = weight bits = bias bits = activation bits; 2*n = multiplier bits; 3*n = accumulator bits
        self.parser.add_argument('--argmax_greater_equal', '--argmax', action='store_true', help='Parameter to call if in final argmax of the logic, a comparison with greater equal should be done instead of just greater')
        self.parser.add_argument('--no_full_logic_report', action='store_true', help='Parameter to call if no full logic report should be created.')
        self.parser.add_argument('--blockwise_logic_report', action='store_true', help='Parameter to call if blockwise logic report should be created, i.e. per logic module.')
        self.parser.add_argument('--no_logic_simulation', '--no_sim', '--no_simulation', action='store_true', help='Parameter to call if the created logic should not be simulated, which might make sense if e.g. just the logic file is wanted or the report should be derived.')
        self.parser.add_argument('--no_logic_test_data_snapshot', action='store_true', help='Parameter to call if the test data should not be snapshotted as numpy arrays for potential future simulations.')

        # Arguments related to LogicNet
        self.parser.add_argument('--lgn_depth', type=int, default=8, help='The number of layers in each LogicNet block.')
        self.parser.add_argument('--lgn_width', type=int, default=1024, help='The number of LUTs in each layer in each LogicNet block.')
        self.parser.add_argument('--lgn_lutsize', type=int, default=6, help='The LUT-size in each LUT in each LogicNet block.')
        self.parser.add_argument('--lgn_lutsize_automatic', '--lutsize_auto', action='store_true', help='Parameter to call if for LogicNet the LUT-Size of each module should automatically be defined by the number of inputs to the module, i.e. ignoring args.lgn_lutsize')
        self.parser.add_argument('--lgn_keep_intermediate_files', action='store_true', help='Parameter to call if for LogicNet all intermediately created files should not be deleted (.data files, flist-file, shell-script, log-files, bit-wise Verilog modules)')

        # Arguments related to Random Forest
        self.parser.add_argument('--rf_estimators', type=int, default=2, help='The number of estimators (decision trees) in the random forest.')
        self.parser.add_argument('--rf_max_depth', type=int, default=5, help='The maximal depth of the trees in the random forest.')
        self.parser.add_argument('--rf_keep_intermediate_files', '--rf_keep', action='store_true', help='Parameter to call if for Random Forest all intermediately created files should not be deleted (bit-wise Verilog modules)')
        self.parser.add_argument('--rf_create_text_files', '--rf_text', action='store_true', help='Parameter to call if for Random Forest the learned rules from the decision trees should be written to file')
        self.parser.add_argument('--rf_no_inverse_weighting', '--rf_no_weight', action='store_true', help='Parameter to call if for Random Forest the under-represented classes should not be upweighted in decision trees.')
        self.parser.add_argument('--rf_no_bitwise_training', '--rf_no_bitwise', action='store_true', help='Parameter to call if for Random Forest the training and logic module derivation should not be done bitwise.')
        self.parser.add_argument('--rf_threshold_upper', '--rf_upper', '--rf_thresh', action='store_true', help='Parameter to call if for a non-bitwise Random Forest training the next higher integer of the float threshold should be chosen (otherwise lower).')

        args = self.parser.parse_args()

        # checking the argparser settings and setting small defaults
        args = self.assertions_and_settings(args)

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.handler_overwrite = not args.no_handler_overwrite
        args.tensorboard_logging = not args.tensorboard_disable
        args.validation = not args.no_validation
        args.bias = not args.no_bias
        args.original_experiment_name = args.experiment_name
        args.lr_scheduling = not args.no_lr_scheduling
        args.rf_inverse_weighting = not args.rf_no_inverse_weighting
        args.balanced_test_data = not args.no_balanced_test_data
        args.full_logic_report = not args.no_full_logic_report
        args.logic_simulation = not args.no_logic_simulation
        args.rf_bitwise_training = not args.rf_no_bitwise_training
        args.logic_test_data_snapshot = not args.no_logic_test_data_snapshot
        args.rf_threshold_lower = not args.rf_threshold_upper

        if args.full_comparison: 
            args.nn_translation = True 
            args.random_forest = True 
            args.logic_net = True 

        # Checking CUDA Settings
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        return args


    # ---------------- Argparser Assertions ----------------
    def assertions_and_settings(self, args):
        '''
            This method checks whether the combination of argparser settings is allowed and valid within this framework - changes some of them.
            Args: 
                args: the argparser arguments 
            Returns: 
                the changed args
        '''

        # --------- checking for valid values -----------
        assert str.lower(args.datareader) in ['coimbra', 'mnist'], 'Invalid Argparser Arguments: Requested DataReader [{}] is not available, choose between the following: [coimbra, mnist]'.format(args.datareader)
        assert str.lower(args.verbosity) in ['debug', 'info'], 'Invalid Argparser Arguments: Verbosity [{}] is not defined - only use one of the following [info, debug].'.format(args.verbosity)
        assert args.test_split < 1.0 and args.test_split > 0.0, 'Invalid Argparser Arguments: Test Split [{}] must be in range (0.0, 1.0)'.format(args.test_split)
        assert args.validation_split < 1.0 and args.validation_split >= 0.0, 'Invalid Argparser Arguments: Validation Split [{}] must be in range [0.0, 1.0)'.format(args.validation_split)
        assert args.unbalanced_class_train_keep < 1.0 and args.unbalanced_class_train_keep >= 0.0, 'Invalid Argparser Arguments: unbalanced_class_train_keep [{}] must be in range [0.0, 1.0)'.format(args.unbalanced_class_train_keep)
        assert args.data_normalization_mode in range(0,8), 'Invalid Argparser Arguments: Data Normalization Mode [{}] is not defined - only use one of the following [0, 1, 2, 3, 4, 5, 6, 7].'.format(args.data_normalization_mode)
        assert str.lower(args.architecture) in ['fcnn'], 'Invalid Argparser Arguments: Architecture [{}] is not defined - only use one of the following [fcnn].'.format(args.architecture)
        assert str.lower(args.hidden_activation) in ['relu', 'htanh'], 'Invalid Argparser Arguments: Hidden Activation [{}] is not defined - only use one of the following [relu, htanh].'.format(args.hidden_activation)
        assert args.epochs > 0, 'Invalid Argparser Arguments: Number of epochs [{}] must be > 0'.format(args.epochs)
        assert str.lower(args.loss) in ['bce', 'default'], 'Invalid Argparser Arguments: Loss [{}] is not defined - only use one of the following [bce, default].'.format(args.loss)
        assert str.lower(args.optimizer) in ['adam', 'sgd'], 'Invalid Argparser Arguments: Optimizer [{}] is not defined - only use one of the following [adam, sgd].'.format(args.optimizer)
        assert str.lower(args.snapshotting_criterion) in ['min_loss', 'max_loss', 'min_accuracy', 'max_accuracy'], 'Invalid Argparser Arguments: Val Score [{}] is not defined - only use one of the following [min_loss, max_loss, min_accuracy, max_accuracy].'.format(args.val_score)
        assert str.lower(args.aig_optimization_command) in ['syn2', 'dc2', 'mfs', 'mfs_advanced'], 'Invalid Argparser Arguments: AIG Optimization Command [{}] is not defined - only use one of the following [syn2, dc2, mfs, mfs_advanced].'.format(args.aig_optimization_command)

        # --------- checking for valid combinations -----------
        if args.bnn: 
            args.hidden_activation = 'htanh' # setting the activation function of hidden units to be a htanh and ignore any other chosen setting 
            args.binary_weights = True  # binarized neural network ignore any other chosen setting 

        if args.load_snapshot: # meaning its not None 
            if args.load_snapshot[-1] != '/': # checking for missed out slash
                args.load_snapshot = args.load_snapshot + '/'
        
        if args.just_logic: 
            assert args.logic_net or args.random_forest or args.nn_translation or args.full_comparison or args.random_forest_direct, 'Invalid Argparser Arguments: You tried creating just logic, but did not specify which types of logic (random forest, direct random forest, LogicNet or NN-Translation)'.format(args.datareader)
            args.epochs = 1 # setting to not do training but just the logic creation

        if args.random_forest_direct: 
            assert not args.logic_net and not args.random_forest and not args.nn_translation and not args.full_comparison, 'Invalid Argparser Arguments: Direct Random Forest is not compatible with LogicNet, Random Forest and NN-Translation simultaneously. Do either one or the other.'

        if args.random_forest or args.logic_net or args.nn_translation or args.random_forest_direct:
            assert (args.total_bits > args.fractional_bits), 'Invalid Arself.parser arguments: for a valid quantization, the number of total bits [{}] has to bigger than the number of fractional bits [{}].'.format(args.total_bits, args.fractional_bits)
            assert (args.fractional_bits >= 1), 'Invalid Arself.parser arguments: for a valid quantization, the number of fractional bits [{}] has to be at least 1.'.format(args.fractional_bits)
            args.hidden_activation = 'relu' # setting the activation function of hidden units to be a ReLU and ignore any other chosen setting 

        if args.logic_net and not args.lgn_lutsize_automatic: 
            assert args.lgn_lutsize <= min(args.hidden_layer_outnodes), 'Invalid Argparser Arguments: The LogicNet-Lutsize [{}] is not allowed to be greater than the smallest number of nodes in a hidden layer [{}]! Adapt one of the argparser arguments: lgn_lutsize or hidden_layer_outnodes!'.format(args.lgn_lutsize, min(args.hidden_layer_outnodes))

        if args.stochastic_binarization: 
            assert args.binary_weights and args.hidden_activation == 'htanh', 'Invalid Argparser Arguments: Stochastic Binarization can only be used with binary weights and the htanh-activation function. Additionally use: --bnn'
        return args


    def default_handling(self, args):
        '''
            This sets a few defaults for other argparser arguments that are needed for a correct execution within the framework. 
            Args: 
                args: the argparser arguments 
            Returns: 
                the modified and corrected args
        '''

        if args.random_forest or args.logic_net or args.nn_translation: 
            optim_flag = False # handling that the AIG optimization command is explicitly used but the optimization is not enabled (which is probably unwanted)
            
            for a in self.used_args: 
                if ('--aig_optimization_command' in a) or ('--aig_optim_cmd' in a): 
                    optim_flag = True 
            
            if optim_flag: 
                args.aig_optimization = True

        # the default parameters in the argparser arguments that relate to the datareader are only set for the case where the datareader is for the coimbra dataset
        # therefore choose other defaults here, if the gi-bleeding or MNIST data set was chosen and the parameter is not explicitly called 

        if str.lower(args.datareader) == 'mnist':

            nodes_flag = True # handling the neural network default structure for MNIST
            act_flag = True # handling activation function as a default - needs to be ReLU, as the rest is not implemented in a logic conversion
            weights_flag = True # handling real-valued weights as default
            bs_flag = True # handling batch size default 
            norm_flag = True # handling the data normalization as a default 
            lr_flag = True # handling learning rate as a default 
            
            for a in self.used_args: 
                if '--hidden_layer_outnodes' in a or '--nodes' in a: 
                    nodes_flag = False 
                if ('--hidden_activation' in a): 
                    act_flag = False 
                if '--batch_size' in a or '--bs' in a: 
                    bs_flag = False
                if '--data_normalization_mode' in a or '--norm' in a: 
                    norm_flag = False 
                if '--lr' in a: 
                    lr_flag = False 

            if nodes_flag: 
                args.hidden_layer_outnodes = [48, 24]
                # args.hidden_layer_outnodes = [1200, 1200] # alternative default structure
                # args.hidden_layer_outnodes = [128, 32] # alternative default structure from tensorflow example (with learning rate 0.01 and 2000 epochs, batch size 100)
                # args.hidden_layer_outnodes = [200, 100, 60, 30, 10] # alternative default for ReLU act and 5 layer fc from tensorflow
            if act_flag and not args.bnn and not args.stochastic_binarization: 
                args.hidden_activation = 'relu'
            if bs_flag: 
                args.batch_size = 100 
            if norm_flag: 
                args.data_normalization_mode = 0 # no normalization
            if lr_flag: 
                args.lr = 0.01 

        # TODO make correct for other data sets 
        # TODO: check that for loading a snapshot, also the task is the same as previously??? 
        # TODO: feature exclusion in GI-Bleeding Dataset ? 
        # TODO: other assertions in terms of loss function for binary-classification or multi-class classification
        # TODO: really needed that fractional bits cannot be 0? 

        return args

    def get_parser(self):
        '''
            Returns: 
                the actual argparse.ArgumentParser() class object
        '''
        return self.parser
    
    def get_args(self):
        '''
            Returns: 
                the actual argparser arguments
        '''
        return self.args
    
    def __repr__(self):
        '''
            This method returns a readable string representation of the argparser configurations chosen.
        '''

        rep_str = '------------------- \nExperiment_Name: {} | Total Runs of Setup: {}\n------------------- \n\n'.format(self.args.experiment_name, self.args.num_runs)

        if not self.args.random_forest_direct: 
            rep_str += '\n---- Network Configurations ----\n'
            rep_str += 'Architecture: {}\n'.format(self.args.architecture)
            rep_str += 'Number of hidden layers / blocks: {}\n'.format(len(self.args.hidden_layer_outnodes))
            rep_str += 'Binary Weights: {}\n'.format(self.args.binary_weights)
            if self.args.stochastic_binarization and self.args.binary_weights: 
                rep_str += 'Binarization: Stochastic\n'
            elif self.args.binary_weights: 
                rep_str += 'Binarization: Deterministic\n'
            rep_str += 'Nodes per hidden layer: {}\n'.format(self.args.hidden_layer_outnodes)
            rep_str += 'Epochs: {}\n'.format(self.args.epochs)
            rep_str += 'Initial Learning Rate: {}\n'.format(self.args.lr)
            rep_str += 'Learning Rate Scheduling (Every 40 Epochs): {}\n'.format(self.args.lr_scheduling)
            rep_str += 'Dropout: {}\n'.format(self.args.dropout)
            rep_str += 'Use Sigmoid on Last Layer for Binary Classification: {}\n'.format(self.args.sigmoid)
            rep_str += 'Batch Normalization: {}\n\n'.format(self.args.batchnorm)

        rep_str += '---- Dataset Configurations ----\n'
        rep_str += 'Reader: {}\n'.format(self.args.datareader)
        rep_str += 'Batch Size: {}\n'.format(self.args.batch_size)
        norm = {
            0: 'No Normalization', 
            1: 'Z-Score Normalization', 
            2: 'Min-Max-Scaling', 
            3: 'Max-Abs-Scaling', 
            4: 'Robust-Scaling', 
            5: 'Power-Transforming', 
            6: 'Quantile-Transforming', 
            7: 'Independent Normalization'}
        rep_str += 'Data Normalization Mode: {} ({})\n'.format(self.args.data_normalization_mode, norm[self.args.data_normalization_mode])
        if str.lower(self.args.datareader) == 'mnist': 
            rep_str += 'Using Test Data Set Split from Argparser: {}\n'.format(self.args.mnist_use_args_test_splits) 
        if ((str.lower(self.args.datareader) == 'mnist') and not (self.args.mnist_use_args_test_splits)) or (str.lower(self.args.datareader) != 'mnist'): 
            rep_str += 'Percentage of Test Data From Overall Data: {} %\n'.format(self.args.test_split*100.0)
            rep_str += 'Perfectly Balanced Test Data Among All Classes: {}\n'.format(self.args.balanced_test_data)
            rep_str += 'Percentage of Most Underrepresented Class that is Kept for Training (Unbalanced Training Set): {}\n'.format(self.args.unbalanced_class_train_keep)
        rep_str += 'Percentage of Validation Data From Training Data: {} %\n'.format(self.args.validation_split*100.0)
        rep_str += 'Enforcing New Data Creation: {}\n'.format(self.args.enforce_new_data_creation)
        if not self.args.validation: 
            rep_str += 'Validation Disabled\n\n'
        else: 
            rep_str += '\n'
        
        if self.args.random_forest or self.args.random_forest_direct:
            if self.args.random_forest_direct:
                rep_str += '---- Direct Random Forest Training Settings ----\n'
            else: 
                rep_str += '---- Random Forest Training Settings ----\n'
            rep_str += 'Maximal Depth: {}\n'.format(self.args.rf_max_depth)
            rep_str += 'Number of Estimators: {}\n'.format(self.args.rf_estimators)
            rep_str += 'Create Textfiles from Decision Trees: {}\n'.format(self.args.rf_create_text_files)
            if not self.args.random_forest_direct:
                rep_str += 'Bitwise Training: {}\n'.format(self.args.rf_bitwise_training)
                if not self.args.rf_bitwise_training: 
                    rep_str += 'Lower Integer Threshold: {}\n'.format(self.args.rf_threshold_lower)
            rep_str += 'Inverse Weighting: {}\n'.format(self.args.rf_inverse_weighting)
            rep_str += 'Keep Intermediate Files: {}\n\n'.format(self.args.rf_keep_intermediate_files)

        if self.args.logic_net: 
            rep_str += '---- LogicNet Training Settings ----\n'
            rep_str += 'LogicNet Depth (Number of Layers): {}\n'.format(self.args.lgn_depth)
            rep_str += 'LogicNet Width (Number of LUTs In Each Layer): {}\n'.format(self.args.lgn_width)
            if not self.args.lgn_lutsize_automatic: 
                rep_str += 'LogicNet LUT-Size (In Each LUT): {}\n'.format(self.args.lgn_lutsize)
            else: 
                rep_str += 'LogicNet LUT-Size: Chosen automatically\n'
            rep_str += 'Keep Intermediate Files: {}\n\n'.format(self.args.lgn_keep_intermediate_files)

        if self.args.random_forest or self.args.logic_net or self.args.nn_translation or self.args.random_forest_direct:
            rep_str += '---- Other General Logic Settings ----\n'
            rep_str += 'Total Number of Quantization Bits: {}\n'.format(self.args.total_bits)
            rep_str += 'Number of Fractional Quantization Bits: {}\n'.format(self.args.fractional_bits)
            rep_str += 'AIG Optimization before Statistics and Simulation: {}\n'.format(self.args.aig_optimization)
            rep_str += 'Final Argmax Greater Equal: {}\n'.format(self.args.argmax_greater_equal)
            rep_str += 'AIG Export: {}\n'.format(self.args.aig_export)
            rep_str += 'ABC Verilog Export: {}\n'.format(self.args.abc_verilog_export)
            if not self.args.random_forest_direct:
                rep_str += 'No NN Training and Just Logic Creation: {}\n'.format(self.args.just_logic)
            rep_str += 'Logic Simulation: {}\n'.format(self.args.logic_simulation)
            rep_str += 'Creation of Full Interpretable Logic Report: {}\n'.format(self.args.full_logic_report)
            if not self.args.random_forest_direct:
                rep_str += 'Creation of Blockwise Interpretable Logic Report: {}\n'.format(self.args.blockwise_logic_report)
            rep_str += 'Creation of Test Data Snapshot for Future Logic Simulation: {}\n'.format(self.args.logic_test_data_snapshot)
            rep_str += 'Results of Logic Testing Filename: {}.csv\n\n'.format(self.args.logic_results_filename)


        rep_str += '---- Additional General Settings ----\n'
        rep_str += 'Verbosity Level: {}\n'.format(str.upper(self.args.verbosity))
        rep_str += 'Resource Monitoring: {}\n'.format(self.args.resource_monitoring)
        rep_str += 'Logging Interval: {}\n'.format(self.args.log_interval)
        rep_str += 'Results Subfolder: {}\n'.format(self.args.results_subfolder)
        if not self.args.random_forest_direct:
            rep_str += 'Netron Export: {}\n'.format(self.args.netron_export)
            rep_str += 'Results of NN Testing Filename: {}.csv\n'.format(self.args.nn_results_filename)
            rep_str += 'Tensorboard Logging: {}\n'.format(self.args.tensorboard_logging)
            rep_str += 'CUDA Usage: {} - Seed: {}\n'.format(self.args.cuda, self.args.seed)
            rep_str += 'Snapshotting: {}\n'.format(self.args.snapshotting)
            if self.args.snapshotting: 
                rep_str += 'Snapshotting Overwrite: {}\n'.format(self.args.snapshotting_overwrite)
                rep_str += 'Snapshotting Inverval: {}\n'.format(self.args.snapshotting_interval)
                if self.args.snapshotting_best:
                    rep_str += 'Snapshotting Best: {}\n'.format(self.args.snapshotting_best)
                    rep_str += 'Snapshotting Criterion: {}\n'.format(self.args.snapshotting_criterion)
        else: 
            rep_str += 'Results of Direct RF Testing Filename: {}.csv\n'.format(self.args.direct_rf_results_filename)
        rep_str += '\n'

        return rep_str