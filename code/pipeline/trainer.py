from logicml.code.pipeline.logic_processing import *
from logicml.code.pipeline.nn import * 
from logicml.code.pipeline.utils import *
from logicml.code.pipeline.random_forest import *
from logicml.code.dataloaders.data_readers import *
from logicml.code.pipeline.handler import *
from logicml.code.pipeline.result_reporter import *
from logicml.code.pipeline.logic_simulator import *
from tensorboardX import SummaryWriter
import os
import math
import json

# ---------------- Definition of Trainer Class and Routines ----------------

class Trainer: 
    def __init__(self, handler, args, reader, logic_processor, net, optimizer, criterion):
        '''
            Initializes the Trainer that defines the routines of . 
            Args: 
                handler: handler of type Handler() for logging
                args: ArgParser arguments from run script
                reader: Data Reader object (e.g. Coimbra Reader)
                logic_processor: LogicProcessing object
                net: the neural network object
                optimizer: the optimizer for training the network
                criterion: the loss function of the network
        '''
        if handler is None: 
            self.handler = Handler()
        else: 
            assert isinstance(handler, Handler)
            self.handler = handler
        
        self.args = args
        
        assert isinstance(reader, AbstractDataReader)
        self.reader = reader
        
        assert isinstance(logic_processor, LogicProcessing)
        self.logic_processor = logic_processor
        
        assert isinstance(net, LogicNN)
        self.net = net

        if self.args.cuda: 
            self.net = self.net.cuda()

        assert isinstance(optimizer, torch.optim.Optimizer)
        self.optimizer = optimizer
        
        # TODO: what kind of assertion for criterion? 
        self.criterion = criterion

        if self.args.tensorboard_logging:
            self.tb = SummaryWriter(self.handler.tensorboard_path)
        
        if self.args.validation: 
            # a list with validation loss scores that will be filled during run-time 
            # NOTE: this is used for snapshotting best and latest model 
            # TODO: argparser argument to disble snapshotting? If yes, then this should be handled here 
            self.val_scores = []
            # function for exporting, whether max or min is wanted 
            self.expfct = {'min': np.min, 'max': np.max}[str.lower(self.args.snapshotting_criterion).split('_')[0]]

        self.handler.log('info', 'Started trainer for experiment: {}'.format(self.args.experiment_name))

        if self.args.no_validation: 
            self.handler.log('warning', 'Validation runs are turned of. No snapshots of the trained network will be created.')

        if self.args.snapshotting: 
            if not self.args.snapshotting_overwrite: 
                error_msg = 'Snapshotting cannot be fulfilled because you would overwrite an exisiting snapshot with the same experiment name. If you wish to overwrite, call --snapshotting_overwrite or use a different experiment name!'
                assert not os.path.isdir(self.handler.path('models/{}'.format(self.args.experiment_name), use_results_path=True, no_mkdir=True)), error_msg
            
            # Boolean flag for when the parameters of the network and the reader should be stored, will be set to False during runtime, once it was stored already
            self.snapshot_save_parameters = True

        # check for a snapshot to load, which leads to reconstruction of the network and the reader and ignoring potential other argparser arguments 
        if self.args.load_snapshot: # meaning its not None 
            self.load_snapshot(self.args.load_snapshot)
            if self.args.logic_net and not self.args.lgn_lutsize_automatic:
                error_msg = 'Invalid Argparser Argument after loading snapshot: The LogicNet-Lutsize [{}] is not allowed to be greater than the smallest number of nodes in a hidden layer [{}]!'.format(args.lgn_lutsize, min(args.hidden_layer_outnodes))
                assert args.lgn_lutsize <= min(self.args.hidden_layer_outnodes), error_msg

    def train_nn(self): 
        '''
            Defines the training routine for a neural network, i.e. training the NN weights, processing of data batches, creating logic.
        '''

        # Resource Monitoring
        if self.args.resource_monitoring:
            self.handler.start_resource_monitoring(interval=60)

        # Set network to training mode
        self.net.train()

        # ---------------- ITERATION: over epochs ----------------
        
        # to make sure that in this special case, training is really as long as the declared epochs and translation to logic is done afterwards
        if self.args.random_forest or self.args.logic_net or self.args.nn_translation: 
            max_epochs = self.args.epochs+2
        else: 
            max_epochs = self.args.epochs+1
        
        for epoch in range(1, max_epochs):
            
            if epoch == 1: 
                time_for_network_vis = True # a flag that is used to create the onnx and pth model exports

            # check random forest or logicnet creation or translation of NN to logic standalone
            if self.args.random_forest or self.args.logic_net or self.args.nn_translation: 
                time_for_logic = (epoch == max_epochs - 1)

            else: 
                time_for_logic = False


            # shuffling instances and creating new training batches to not show data in same order
            self.reader.shuffle_and_tie_new_batches()

            # resetting running loss 
            running_loss = 0.0

            # changing learning rate every 40 epochs for adam and sgd 

            # TODO: also make curriculum for logic and in general more options from the outside (with argparser)?? 

            if (epoch%40==0) and self.args.lr_scheduling: 
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.1
                self.net.optimizer = self.optimizer

            # including collection of activations
            # can be due to logic synthesis for or random forest / logicnet training
            if time_for_logic:
                if self.args.random_forest: 
                    self.handler.tic('Random Forest Logic Procedure')
                    self.handler.log('info', 'Train Epoch: {} - Random Forest Logic Creation'.format(epoch))
                if self.args.logic_net: 
                    self.handler.tic('LogicNet Procedure')
                    self.handler.log('info', 'Train Epoch: {} - LogicNet Logic Creation'.format(epoch))
                if self.args.nn_translation: 
                    self.handler.tic('NN Translation Procedure')
                    self.handler.log('info', 'Train Epoch: {} - NN Translation Logic Creation'.format(epoch))

                # set network to evaluation mode because we dont want to optimize the weights
                self.net.eval()

                # following used in blockwise logic creation
                self.accumulated_truthtables_dict = None

            else: 
                self.handler.tic('Main Training Routine')
                self.handler.log('info', 'Train Epoch: {} - Normal Training Routine'.format(epoch))

            # ---------------- ITERATION: over batches  ----------------
            for didx, batch_data in enumerate(self.reader.train_batches):

                # save model for visualization with NETRON (at beginning of training routine)
                if time_for_network_vis and didx == 0 and self.args.netron_export: 
                    folder_path = self.handler.path('netron_exports')
                    file_name = '{}_{}_netronvis'.format(epoch, self.args.experiment_name)
                    self.save_net_for_visualization(batch_data, folder_path=folder_path, file_name=file_name)
                    time_for_network_vis = False

                # add one to didx for right print-outs
                didx += 1

                # collect activations and dont do optimization steps
                if time_for_logic:
                    result_dict = self.net.training_routine(epoch, batch_data, self.optimizer, self.criterion, return_logic_infos=True)
                    acts = result_dict['activation_features'] # list with numpy arrays
                    outs = result_dict['outputs'] # list with numpy arrays 

                    # for blockwise processing - creating truthtables for blocks of logic that can be stacked to model the overall net's logic
                    # each key is the identifier of an output node, values = sequence of list with input node names (column variables), input side arrays, output side arrays 
                    # those truth tables can be used to train RandomForest or LogicNet on it
                    self.accumulated_truthtables_dict = collect_activations_and_outputs_blockwise(self.accumulated_truthtables_dict, acts, outs, self.net.layer_connections_dict)

                else: 
                    # normal training with current network structure and with optimization steps
                    result_dict = self.net.training_routine(epoch, batch_data, self.optimizer, self.criterion)
                
                    # Loss Handling
                    loss = result_dict['loss']
                    running_loss += loss
                    
                    # Logging of batch-wise loss
                    if didx % self.args.log_interval == 0:
                        self.handler.log('debug', 'Train Epoch: {} [Batch: {}/{} ({:.2f}%)] Loss: {:.6f}'.format(epoch, didx, len(self.reader.train_batches), 100. * didx / len(self.reader.train_batches), loss))

                    if self.args.tensorboard_logging:
                        self.tb.add_scalar('99 Detailed/DetailedTrainingLossPerBatch', loss, epoch * len(self.reader.train_batches) + didx)
            
            # ---------------- Neural Network to Logic Standalone Translation Routine (Logic Synthesis) ----------------
            verilog_files_dict = None
            if self.args.nn_translation and time_for_logic: 
                folder_path = self.handler.path('nn_logic_files')
                # creating a Verilog module file for each node in the NN 
                verilog_files_dict = self.logic_processor.neural_net_dump_verilog(self.net, folder_path, suffix='nn', only_first_layer=False)
                self.handler.toc('NN Translation Procedure')
            
            if not self.args.nn_translation and time_for_logic: 
                folder_path = self.handler.path('nn_logic_files')
                # means that the test_logic() method will be called from random forest or LogicNet option
                # therefore create only Verilog modules for each node of the first layer of the NN
                verilog_files_dict = self.logic_processor.neural_net_dump_verilog(self.net, folder_path, suffix='nn', only_first_layer=True)

            # create interpretable report for each NN logic module that was created if this option was chosen
            if self.args.blockwise_logic_report and verilog_files_dict: 
                report_folder_path = self.handler.path('blockwise_reports/nn_logic_files')
                for k, verilog_file in verilog_files_dict.items(): 
                    if 'l0' not in k: # hidden and output layers
                        names, ins, outs = self.accumulated_truthtables_dict[k]
                        in_key = k+'_nn_x'
                    else: # first layer
                        names = self.reader.feature_names
                        in_key = 'final_in'
                    self.logic_processor.logic_to_interpretable_equation_report(verilog_file, names, k+'_nn_activation', report_folder_path, k, out_key=k+'_nn_out', in_key=in_key)
                    self.handler.log('info', 'NN Translation Logic Block - {}: Created interpretable logic report.'.format(k))


            # ---------------- Random Forest Training and Translation Routine (Logic Synthesis) ----------------
            if self.args.random_forest and time_for_logic: 
                
                folder_path = self.handler.path('random_forest_files')

                # a list that saves the files that are only created for computational purposes and can be deleted later
                files_to_delete = []

                for k in self.accumulated_truthtables_dict.keys():

                    # TODO: here we do a training of random forests bit by bit - maybe adapt this to training with blocks of bits ?? 

                    self.handler.log('info', 'Random Forest - Processing Logic Block: {}'.format(k))
                    names, ins, outs = self.accumulated_truthtables_dict[k]
                    
                    # STARTING BITWISE TRAINING
                    if self.args.rf_bitwise_training: 
                        # train a logic net per bit
                        for bit_idx in range(self.args.total_bits): 
                            self.handler.log('debug', 'Random Forest - Processing Logic Block: {} - Bit: {}'.format(k, self.args.total_bits-bit_idx-1))
                            ins_bit_arr = np.array([ins[i][j][bit_idx] for j in range(ins.shape[1]) for i in range(ins.shape[0])]).reshape(*ins.shape)
                            ins_bit_arr = np.array(ins_bit_arr, dtype=np.float)
                            outs_bit_arr = np.array([outs[i][bit_idx] for i in range(outs.shape[0])]).reshape(*outs.shape)
                            outs_bit_arr = np.array(outs_bit_arr, dtype=np.float)

                            # NOTE: file name only includes the outputting node as we only consider fully connected architecture
                            # TODO: change once we also allow skip connections
                            # NOTE: a renaming: in Logic, the MSB has the highest bit_number, but here in the for-loop, the lowest bit_idx refers to the MSB
                            file_name = k + '_rf_bit{}'.format(self.args.total_bits-bit_idx-1)

                            # NOTE: here we have a problem: 
                            # when under all labels, only one label type occurs, the random forest creates just one output class 
                            # then we get an error on our assertion that we have 2 classes within one of the random forest methods
                            # in this case, the logic implementation of the random forest would anyway just always be the one label for each input combination
                            # i.e. we need to treat this case differently and do not even need to train a random_forest, but just write a logic, that takes inputs, doesnt use them and does a constant label output
                            if len(np.unique(outs_bit_arr)) == 1: 
                                # NOTE: int conversion of the value only because in the numpy array the type could be np.uint8 which is not the same as python int
                                random_forest_substitute_verilog_bitwise(ins_bit_arr.shape[1], 1, 1, int(np.unique(outs_bit_arr)[0]), folder_path, file_name)
                                self.handler.log('debug', 'Random Forest: Needed to write substitute Verilog module for logic block {}'.format(k))

                            else: 
                                # TODO: maybe write print_out_msg to log_file because it tells something about training performance of this method for further module stacking
                                model, print_out_msg = generate_forest(file_name, ins_bit_arr, outs_bit_arr, self.args.rf_estimators, self.args.rf_max_depth, verification=False, inverse_weighting=self.args.rf_inverse_weighting)

                                # option to "visualize" the decision trees in form of text-files that are created to read out the set of rules
                                if self.args.rf_create_text_files: 
                                    write_random_forest_to_txt_file(model, self.handler.path('random_forest_files/text_files'), 'rf_{}_bit{}'.format(k, bit_idx), feature_names=names)

                                self.handler.log('debug', 'Random Forest Training Result: {}'.format(print_out_msg))
                                random_forest_dump_verilog_bitwise(model, ins_bit_arr.shape[1], 2, self.args.total_bits, self.args.fractional_bits, 1, 1, folder_path, file_name)

                            # adding the intermediate bit-wire verilog file to the list of files that should be deleted
                            files_to_delete.append(folder_path+file_name+'.v')

                        # concatenate the random forest per bit modules here and turn them into one LogicNet block per outputting node
                        verilog_file = self.logic_processor.concatenate_random_forest_bitwise_verilog_modules(len(names), folder_path, k+'_rf')

                    else: 
                        # TODO: maybe write print_out_msg to log_file because it tells something about training performance of this method for further module stacking

                        # NOTE: at this point ins and outs arrays are already the integer representations of the binary binary strings from the quantization scheme
                        model, print_out_msg = generate_forest(k, np.array(ins, dtype=np.float), np.array(outs, dtype=np.float), self.args.rf_estimators, self.args.rf_max_depth, verification=False, inverse_weighting=self.args.rf_inverse_weighting)

                        # option to "visualize" the decision trees in form of text-files that are created to read out the set of rules
                        if self.args.rf_create_text_files: 
                            write_random_forest_to_txt_file(model, self.handler.path('random_forest_files/text_files'), 'rf_{}'.format(k), feature_names=names)

                        self.handler.log('debug', 'Random Forest Training Result: {}'.format(print_out_msg))
                        
                        verilog_file = random_forest_to_verilog_direct(model, ins.shape[1], self.args.total_bits, self.args.fractional_bits, k+'_rf', folder_path, threshold_lower=self.args.rf_threshold_lower)

                    # create interpretable report for the logic module if this option was chosen
                    if self.args.blockwise_logic_report and self.args.rf_bitwise_training: 
                        report_folder_path = self.handler.path('blockwise_reports/random_forest_files')
                        self.logic_processor.logic_to_interpretable_equation_report(verilog_file, names, k+'_rf_activation', report_folder_path, k, in_key=k+'_rf_x', out_key=k+'_rf_out')
                        self.handler.log('info', 'Random Forest Logic Block - {}: Created interpretable logic report.'.format(k))

                # delete the bit-wise Verilog files once they are concatenated
                # the dumps of each verilog file are kept, due to potential stacking of modules from different techniques later on
                if not self.args.rf_keep_intermediate_files: 
                    for f in files_to_delete: 
                        os.system('rm {}'.format(f))
                    
                self.handler.toc('Random Forest Procedure')

            # ---------------- LogicNet Training and Translation Routine (Logic Synthesis) ----------------
            if self.args.logic_net and time_for_logic: 

                folder_path = self.handler.path('logicnet_files')

                # a list that saves the files that are only created for computational purposes and can be deleted later
                files_to_delete = []

                for k in self.accumulated_truthtables_dict.keys():
                    self.handler.log('info', 'LogicNet - Processing Logic Block: {}'.format(k))
                    names, ins, outs = self.accumulated_truthtables_dict[k]

                    # train a logic net per bit
                    for bit_idx in range(self.args.total_bits): 
                        self.handler.log('debug', 'LogicNet - Processing Logic Block: {} - Bit: {}'.format(k, self.args.total_bits-bit_idx-1))
                        ins_bit_arr = np.array([ins[i][j][bit_idx] for j in range(ins.shape[1]) for i in range(ins.shape[0])]).reshape(*ins.shape)
                        ins_bit_arr = np.array(ins_bit_arr, dtype=np.uint8)
                        outs_bit_arr = np.array([outs[i][bit_idx] for i in range(outs.shape[0])]).reshape(*outs.shape)
                        outs_bit_arr = np.array(outs_bit_arr, dtype=np.uint8)

                        # TODO: also introduce a substituting logic for LogicNet if len(np.unique(outs_bit_arr)) == 1 ?? 

                        # NOTE: file name only includes the outputting node as we only consider fully connected architecture
                        # TODO: change once we also allow skip connections
                        # NOTE: a renaming: in Logic, the MSB has the highest bit_number, but here in the for-loop, the lowest bit_idx refers to the MSB
                        file_name = k + '_lgn_bit{}'.format(self.args.total_bits-bit_idx-1)

                        # write the .data files that LogicNet needs 
                        # NOTE: LogicNet is always trained on just one bit data: therefore if higher bit activation quantization, then we concatenate multiple LogicNets
                        # TODO: adapt the following procedure once we have more than one bit quantization
                        features_path = write_features_to_logicnet_data_files(ins_bit_arr, 1, folder_path, '{}'.format(file_name)) 
                        labels_path = write_labels_to_logicnet_data_files(outs_bit_arr, 1, folder_path, '{}_labels'.format(file_name))
                        files_to_delete.extend([features_path, labels_path])

                        # write the flist file that LogicNet needs 
                        flist_path = write_logicnet_flist(features_path, labels_path, folder_path, file_name)
                        files_to_delete.append(flist_path)

                        # write the shell script that executes LogicNet on that
                        if self.args.lgn_lutsize_automatic: 
                            lutsize=len(names) # means LUT-Size is automatically chosen depending on the number of inputs to the logic module
                        else: 
                            lutsize=None # means that always self.args.lut_size is chosen
                            
                        shell_scripth_path = self.logic_processor.write_logicnet_executing_shell_script(flist_path, folder_path, file_name, lutsize=lutsize)
                        files_to_delete.append(shell_scripth_path)

                        # call the shell script to actually run LogicNet and dump out the Verilog file 
                        os.system('source {}'.format(shell_scripth_path))

                        # adding the logfile and the intermediate bit-wire verilog file to the list of files that should be deleted
                        files_to_delete.extend([folder_path+file_name+'.v', folder_path+file_name+'.log'])

                    # concatenate the LogicNet per bit modules here and turn them into one LogicNet block per outputting node
                    verilog_file = self.logic_processor.concatenate_logicnet_bitwise_verilog_modules(len(names), folder_path, k+'_lgn')

                    # create interpretable report for the logic module if this option was chosen
                    if self.args.blockwise_logic_report: 
                        report_folder_path = self.handler.path('blockwise_reports/logicnet_files')
                        self.logic_processor.logic_to_interpretable_equation_report(verilog_file, names, k+'_lgn_activation', report_folder_path, k, in_key=k+'_lgn_x', out_key=k+'_lgn_out')
                        self.handler.log('info', 'LogicNet Logic Block - {}: Created interpretable logic report.'.format(k))

                # delete the .data files, the shell scripts and the flist files, as they will anyway not be needed anymore
                # the dumps of each verilog file are kept, due to potential stacking of modules from different techniques later on
                if not self.args.lgn_keep_intermediate_files: 
                    for f in files_to_delete: 
                        os.system('rm {}'.format(f))

                self.handler.toc('LogicNet Procedure')

            # ---------------- End of Epoch for Normal Training Routine ----------------
            elif not time_for_logic:
                # Running Loss Handling 
                running_loss = running_loss / float(len(self.reader.train_batches))
                
                # Logging of running loss
                self.handler.log('info', 'Train Epoch: {} | Running Loss {:.6f}'.format(epoch, running_loss))

                if self.args.tensorboard_logging:
                    self.tb.add_scalar('00 General/TrainingRunningLoss', running_loss, epoch)

                if self.args.validation:
                    val_result_dict = self.validate(epoch)

                    # filling self.val_scores for snapshotting 
                    if 'accuracy' in str.lower(self.args.snapshotting_criterion):
                        self.val_scores.append(val_result_dict['accuracy'])
                    else: # loss 
                        self.val_scores.append(val_result_dict['loss'])
                    
                    # snapshotting of best and latest model according to a snapshotting interval
                    if self.args.snapshotting: 
                        current_best = self.save_snapshot(epoch) 
                        if current_best: 
                            self.handler.log('info', 'Snapshotted a new best network!')
                        # TODO: need of also snapshotting what the validation value was that lead to the best network so far???  

                self.handler.toc('Main Training Routine')

        # Resource Monitoring
        if self.args.resource_monitoring:
            # processing time-log events that were registered with tic toc during training
            self.handler.write_timelog(self.args.experiment_name + '_timelog')
            self.handler.stop_resource_monitoring('training_resources')

    def train_direct_rf(self): 
        '''
            Defines the training routine and logic derivation for a direct random forest on the data.
            NOTE: no support of bitwise training
            
            Returns: 
                the random forest sklearn model
        '''
        self.handler.tic('Direct Random Forest Procedure')
        self.handler.log('info', 'Started training of Direct Random Forest.')

        folder_path = self.handler.path('direct_rf_files')

        # ------------ Creating quantized version of the training data (features and labels) ------------ 

        train_features_quantized = np.zeros(self.reader.train_features.shape, dtype=np.uint8)
        # NOTE: the classes need to be written to the logic as python unsigned binary string
        # because the logic simulator will in the end reverse it again with Python-internal method
        train_labels_quantized = np.array(self.reader.train_labels, dtype=np.uint8)
        train_labels_quantized = train_labels_quantized.reshape(-1)
        # iterate over all possible indices and apply the quantization
        for idx in np.ndindex(*self.reader.train_features.shape):
            #  getting the int representation
            train_features_quantized[idx] = convert_float_to_quantized_int_repr(float(self.reader.train_features[idx]), self.args.fractional_bits, self.args.total_bits)

        # ------------ Creating quantized version of the validation data (features and labels) ------------ 

        validation_features_quantized = np.zeros(self.reader.validation_features.shape, dtype=np.uint8)
        validation_labels_quantized = np.array(self.reader.validation_labels, dtype=np.uint8)
        validation_labels_quantized = validation_labels_quantized.reshape(-1)
        # iterate over all possible indices and apply the quantization
        for idx in np.ndindex(*self.reader.validation_features.shape):
            #  getting the int representation
            validation_features_quantized[idx] = convert_float_to_quantized_int_repr(float(self.reader.validation_features[idx]), self.args.fractional_bits, self.args.total_bits)

        # ------------ Starting model training ------------ 
        file_name = 'direct_rf'

        model, print_out_msg, train_acc, val_acc = generate_forest(file_name, np.array(train_features_quantized, dtype=np.float), np.array(train_labels_quantized, dtype=np.float), self.args.rf_estimators, self.args.rf_max_depth, validation_features=validation_features_quantized, validation_labels=validation_labels_quantized, verification=False, inverse_weighting=self.args.rf_inverse_weighting, additional_return=True)
        self.handler.log('debug', 'Random Forest Training Result: {}'.format(print_out_msg))

        # ------------ Model to text files ------------ 
        # option to "visualize" the decision trees in form of text-files that are created to read out the set of rules
        if self.args.rf_create_text_files: 
            write_random_forest_to_txt_file(model, self.handler.path('direct_rf_files_text_files'), file_name, feature_names=self.reader.feature_names)
        
        # ------------ Starting Verilog file creation ------------ 
        # defining the width of the final output
        out_width = math.ceil(math.log2(self.reader.num_classes + self.reader.num_classes%2))
        verilog_file = random_forest_to_verilog_direct(model, train_features_quantized.shape[1], self.args.total_bits, self.args.fractional_bits, file_name, folder_path, threshold_lower=self.args.rf_threshold_lower, direct_rf=True, direct_rf_out_width=out_width)

        self.handler.toc('Direct Random Forest Procedure')
        # TODO: resource management in trainer? 
        return model

    def test_direct_rf(self, model): 
        '''
            Defines the testing routine of the sklearn direct random forest model on the test data.
            Args: 
                model: sklearn random forest model
        '''

        # ------------ Creating quantized version of the test data (features and labels) ------------ 

        test_features_quantized = np.zeros(self.reader.test_features.shape, dtype=np.uint8)
        test_labels_quantized = np.array(self.reader.test_labels, dtype=np.uint8)
        test_labels_quantized = test_labels_quantized.reshape(-1)
        # iterate over all possible indices and apply the quantization
        for idx in np.ndindex(*self.reader.test_features.shape):
            #  getting the int representation
            test_features_quantized[idx] = convert_float_to_quantized_int_repr(float(self.reader.test_features[idx]), self.args.fractional_bits, self.args.total_bits)

        predictions = model.predict(test_features_quantized)
        predictions = np.array(predictions, dtype=np.uint8)

        # deriving the confusion matrix and reporting the measures
        confusion_matrix = confusion_matrix_from_preds_and_labels(self.reader.num_classes, predictions, test_labels_quantized)
        reporter = ResultReporter(confusion_matrix)
        results_dict = reporter.getResultDict()
        accuracy = results_dict['accuracy'] * 100.0

        # plotting the confusion matrix
        sorted_labels = [*self.reader.class_labels_dict.values()]
        conf_folder_path = self.handler.path('confusion_matrices/{}'.format(self.args.experiment_name), use_results_path=True)
        title = '{} - Direct RF Test'.format(self.args.experiment_name)
        reporter.plot_confmat(sorted_labels, title, conf_folder_path, title)

        # Logging
        self.handler.log('info', 'Direct Random Forest Test | Final Result | Accuracy: ({:.2f}%)'.format(accuracy))
        
        # writing result to CSV file
        results_list = ['{:.2f}'.format(accuracy), \
                '{:.4f}'.format(results_dict['logFScoreSum']), '{:.4f}'.format(results_dict['logFScoreMean']), '{:.4f}'.format(results_dict['precisionMacro']), '{:.4f}'.format(results_dict['precisionMicro']), \
                '{:.4f}'.format(results_dict['recallMacro']), '{:.4f}'.format(results_dict['recallMicro']), '{:.4f}'.format(results_dict['fScoreMacro']), '{:.4f}'.format(results_dict['fScoreMicro'])]

        # if the file doesn't exist yet, we first create it and write the column names to the file 
        if not self.handler.check_results_csv_file_path_exists(self.args.experiment_name, subfolder=self.args.results_subfolder, file_name=self.args.direct_rf_results_filename): 
            column_names = ['Timestamp', 'Experiment Name', 'Accuracy', 'LogFScoreSum', 'LogFScoreMean', 'PrecisionMacro', 'PrecisionMicro', 'RecallMacro', 'RecallMicro', 'fScoreMacro', 'fScoreMicro']
            self.handler.write_result(column_names, self.args.experiment_name, subfolder=self.args.results_subfolder, file_name=self.args.direct_rf_results_filename, writing_column_names=True)
        
        self.handler.write_result(results_list, self.args.experiment_name, subfolder=self.args.results_subfolder, file_name=self.args.direct_rf_results_filename)
        


    def test_logic(self): 
        '''
            Defines the testing routine of a final logic.
        '''

        # Information for saving results
        file_name_suffix = 'final'
        final_logic_dict = {} # keys = file_names, values = paths to the file_name

        # ---------------- Final Verilog File Creation: Concatenating Verilog Modules ----------------

        # NOTE: a standalone logic for each of the called techniques will be created and tested in the following and optionally a combined logic
        # TODO: all of the following methods do not support any skip connections yet - adapt if needed!  

        if self.args.nn_translation: 
            self.handler.tic('NN Logic - Module Concatenation') # resource management
            # doing the concatenation of all the NN Verilog modules
            file_name = 'nn_' + file_name_suffix
            folder_path = self.handler.path('final_logic/{}'.format(file_name))
            self.handler.log('info', 'Logic Test - {}: Concatenating Verilog Modules'.format(file_name))
            final_verilog_file_path = self.logic_processor.concatenate_nn_verilog_modules(self.reader.num_features, self.net.hidden_out_nodes[0], self.accumulated_truthtables_dict, folder_path, file_name)
            final_logic_dict[file_name] = final_verilog_file_path
            self.handler.toc('NN Logic - Module Concatenation') # resource management

        if self.args.random_forest: 
            self.handler.tic('Random Forest - Module Concatenation') # resource management
            # doing the concatenation of all the random forest Verilog modules
            # NOTE: the first layer nodes are anyway modeled from the neural network and the corresponding Verilog modules were already created
            file_name = 'randomforest_' + file_name_suffix
            folder_path = self.handler.path('final_logic/{}'.format(file_name))
            self.handler.log('info', 'Logic Test - {}: Concatenating Verilog Modules'.format(file_name))
            final_verilog_file_path = self.logic_processor.concatenate_random_forest_verilog_modules(self.reader.num_features, self.net.hidden_out_nodes[0], self.accumulated_truthtables_dict, folder_path, file_name)
            final_logic_dict[file_name] = final_verilog_file_path
            self.handler.toc('Random Forest - Module Concatenation') # resource management

        if self.args.logic_net: 
            self.handler.tic('LogicNet - Module Concatenation') # resource management
            # doing the concatenation of all the LogicNet Verilog modules
            # NOTE: the first layer nodes are anyway modeled from the neural network and the corresponding Verilog modules were already created
            file_name = 'logicnet_' + file_name_suffix
            folder_path = self.handler.path('final_logic/{}'.format(file_name))
            self.handler.log('info', 'Logic Test - {}: Concatenating Verilog Modules'.format(file_name))
            final_verilog_file_path = self.logic_processor.concatenate_logicnet_verilog_modules(self.reader.num_features, self.net.hidden_out_nodes[0], self.accumulated_truthtables_dict, folder_path, file_name)
            final_logic_dict[file_name] = final_verilog_file_path
            self.handler.toc('LogicNet - Module Concatenation') # resource management

        if self.args.random_forest_direct: 
            file_name = 'direct_randomforest' + file_name_suffix
            final_logic_dict[file_name] = self.handler.path('direct_rf_files') + 'direct_rf.v'
        
        # TODO: treat combination of modules that might be given from the outside (i.e.) manually - but how? 
        # TODO: otherwise extract information about performance of each module according to method and stack the best ones?  

        # ---------------- Snapshotting the test data for potential future logic simulation runs (e.g. data) ----------------
        if self.args.logic_test_data_snapshot: 
            folder_path = self.handler.path('logic_test_data')
            np.save(folder_path + 'features', self.reader.test_features)
            np.save(folder_path + 'labels', self.reader.test_labels)
            self.handler.log('info', 'Logic Test: Saved test data snapshot for future simulations.')
      
        # ---------------- Iteration over all of the created logic files ----------------
        for file_name, pre_final_verilog_file_path in final_logic_dict.items(): 

            folder_path = self.handler.path('final_logic/{}'.format(file_name))

            # ---------------- ABC: Translating Verilog Logic into AIG and Evaluating ----------------
            # resource management
            if 'logicnet' in file_name: 
                self.handler.tic('LogicNet -  AIG Translation')
            elif 'randomforest' in file_name: 
                self.handler.tic('Random Forest - AIG Translation')
            elif 'nn' in file_name: 
                self.handler.tic('NN Logic - AIG Translation')

            self.handler.log('info', 'Logic Test - {}: Translating Verilog Into AIG'.format(file_name))
            # read the final Verilog file in ABC, get the statistics and dump out the AIG-file, the processed Verilog-file and the eqn-file for simulation
            final_aig_script, aig_statistics_pth, eqn_path = self.logic_processor.write_verilog_to_aig_shell_script(pre_final_verilog_file_path, folder_path, file_name, export_aig=self.args.aig_export)
            os.system('source {}'.format(final_aig_script))
            # remove the created shell script, once it was executed: 
            os.system('rm {}'.format(final_aig_script))

            # resource management
            if 'logicnet' in file_name: 
                self.handler.toc('LogicNet -  AIG Translation')
            elif 'randomforest' in file_name: 
                self.handler.toc('Random Forest - AIG Translation')
            elif 'nn' in file_name: 
                self.handler.toc('NN Logic - AIG Translation')
            
            # ---------------- Other: Information Processing ----------------
            # deriving the intepretable report from the full logic
            if self.args.full_logic_report: 
                self.logic_processor.logic_to_interpretable_equation_report(eqn_path, self.reader.feature_names, 'final_class_prediction', folder_path, file_name)
                self.handler.log('info', 'Logic Test - {}: Created interpretable logic report.'.format(file_name))

            # ---------------- Simulation of the Final Logic ----------------
            if self.args.logic_simulation: 
                self.handler.log('info', 'Logic Test - {}: Starting Simulation'.format(file_name))

                # resource management
                if 'logicnet' in file_name: 
                    self.handler.tic('LogicNet - Simulation')
                elif 'randomforest' in file_name: 
                    self.handler.tic('Random Forest - Simulation')
                elif 'nn' in file_name: 
                    self.handler.tic('NN Logic - Simulation')

                # the actual simulation          
                simulator = LogicSimulator(self.handler, self.args, self.reader.test_features, data_labels=self.reader.test_labels, folder_path=folder_path)
                simulator.add_logic_experiment(eqn_path)
                result_file_path = simulator.simulate()[0] # this method returns a list, but we know that there is only one element inside, because we don't simulate logic for intermediate signals

                # resource management
                if 'logicnet' in file_name: 
                    self.handler.toc('LogicNet - Simulation')
                elif 'randomforest' in file_name: 
                    self.handler.toc('Random Forest - Simulation')
                elif 'nn' in file_name: 
                    self.handler.toc('NN Logic - Simulation')

                # calculating the actual performance of the logic (getting a dictionary with multiple parameters)
                # the calculation of the performance measure is based on the simulation results that were written to a text_file
                confusion_matrix = simulator.calculate_confmat_from_simulation_results(result_file_path, self.reader.num_classes)

                reporter = ResultReporter(confusion_matrix)
                results_dict = reporter.getResultDict()
                accuracy = results_dict['accuracy'] * 100.0

                # plotting the confusion matrix
                sorted_labels = [*self.reader.class_labels_dict.values()]
                conf_folder_path = self.handler.path('confusion_matrices/{}'.format(self.args.experiment_name), use_results_path=True)
                title = '{} - Logic Simulation: {}'.format(self.args.experiment_name, file_name)
                reporter.plot_confmat(sorted_labels, title, conf_folder_path, title)
            
                # getting the statistics of the AIG
                aig_statistics_dict = None
                with open(aig_statistics_pth) as f: 
                    aig_statistics_dict = json.load(f)
                
                # writing result to CSV file
                results_list = [aig_statistics_dict['input'], aig_statistics_dict['output'], aig_statistics_dict['and'], aig_statistics_dict['level'], '{:.2f}'.format(accuracy), \
                    '{:.4f}'.format(results_dict['logFScoreSum']), '{:.4f}'.format(results_dict['logFScoreMean']), '{:.4f}'.format(results_dict['precisionMacro']), '{:.4f}'.format(results_dict['precisionMicro']), \
                    '{:.4f}'.format(results_dict['recallMacro']), '{:.4f}'.format(results_dict['recallMicro']), '{:.4f}'.format(results_dict['fScoreMacro']), '{:.4f}'.format(results_dict['fScoreMicro'])]
                
                if not self.handler.check_results_csv_file_path_exists(self.args.experiment_name + '_' + file_name, subfolder=self.args.results_subfolder, file_name=self.args.logic_results_filename): 
                    # if the file doesn't exist yet, we first create it and write the column names to the file 
                    column_names = ['Timestamp', 'Experiment Name', 'AIG Inputs', 'AIG Outputs', 'AIG ANDs', 'AIG Levels', 'Accuracy', 'LogFScoreSum', 'LogFScoreMean', 'PrecisionMacro', 'PrecisionMicro', \
                        'RecallMacro', 'RecallMicro', 'fScoreMacro', 'fScoreMicro']
                    self.handler.write_result(column_names, self.args.experiment_name + '_' + file_name, subfolder=self.args.results_subfolder, file_name=self.args.logic_results_filename,  writing_column_names=True)

                self.handler.write_result(results_list, self.args.experiment_name + '_' + file_name, subfolder=self.args.results_subfolder, file_name=self.args.logic_results_filename)
                self.handler.log('info', 'Logic Test - {} | Final Result | AIG Inputs: {}, AIG Outputs: {}, AIG ANDs: {}, AIG Levels: {}, Accuracy: {})'.format(file_name, aig_statistics_dict['input'], aig_statistics_dict['output'], aig_statistics_dict['and'], aig_statistics_dict['level'], accuracy))


    def validate(self, epoch): 
        '''
            Defines the validation routine of a neural network.
            Args: 
                epoch: the current epoch (when called from the train()-method)
        '''

        # Set network to testing mode
        self.net.eval()

        # initialization
        running_val_loss = 0
        running_correct = 0

        # ---------------- ITERATION: over batches ----------------
        for didx, batch_data in enumerate(self.reader.validation_batches):
            # add one to didx for right print-outs
            didx += 1

            # passing batch through net 
            result_dict = self.net.testing_routine(batch_data, self.criterion)
            
            # evaluating results
            val_loss = result_dict['loss']
            running_val_loss += val_loss
            correct = result_dict['correct']
            running_correct += correct

            if self.args.tensorboard_logging:
                self.tb.add_scalar('99 Detailed/DetailedValidationLossPerBatch', val_loss, epoch * len(self.reader.validation_batches) + didx)

            # Logging 
            self.handler.log('debug', 'Validation Epoch {}: [Batch: {}/{} ({:.2f}%)] | \tAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, didx, len(self.reader.validation_batches), 100. * didx / len(self.reader.validation_batches), val_loss, correct, self.args.batch_size, 100. * correct / self.args.batch_size))

        # Logging
        accuracy = 100. * running_correct / (len(self.reader.validation_batches)*self.args.batch_size)
        running_val_loss /= len(self.reader.validation_batches)
        self.handler.log('info', 'Validation Epoch: {} | Average Loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, running_val_loss, running_correct, len(self.reader.validation_batches)*self.args.batch_size, accuracy))
        
        if self.args.tensorboard_logging:
            self.tb.add_scalar('00 General/ValidationRunningLoss', running_val_loss, epoch)
            self.tb.add_scalar('00 General/ValidationRunningAccuracy', accuracy, epoch)

        self.net.train() # set back to training mode as default  
        return {'loss': running_val_loss, 'accuracy': accuracy} # return for the purpose of snapshotting


    def test(self, epoch=None): 
        '''
            Defines the testing routine of a neural network.
            Args: 
                epoch: usually None
        '''

        # Set network to testing mode
        self.net.eval()

        # initialization
        running_test_loss = 0
        running_correct = 0

        # lists that collect the predictions and labels of each observation
        # will be filled and turned into numpy array to derive the confusion matrix
        labels = []
        predictions = []

        # ---------------- ITERATION: over batches ----------------
        for didx, batch_data in enumerate(self.reader.test_batches):
            # add one to didx for right print-outs
            didx += 1

            # passing batch through net 
            result_dict = self.net.testing_routine(batch_data, self.criterion)
            
            # evaluating results
            test_loss = result_dict['loss']
            running_test_loss += test_loss
            correct = result_dict['correct']
            running_correct += correct
            labels.extend(result_dict['labels'])
            predictions.extend(result_dict['predictions'])

            if self.args.tensorboard_logging:
                if epoch is not None: 
                    self.tb.add_scalar('99 Detailed/DetailedTestLossPerBatch', test_loss, epoch * len(self.reader.test_batches) + didx)
                else: 
                    self.tb.add_scalar('99 Detailed/DetailedTestLossPerBatch', test_loss, didx)

            # Logging 
            self.handler.log('debug', 'Test{}: [Batch: {}/{} ({:.2f}%)] | Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(' - Epoch {}'.format(epoch) if epoch is not None else '',didx, len(self.reader.test_batches), 100. * didx / len(self.reader.test_batches), test_loss, correct, self.args.batch_size, 100. * correct / self.args.batch_size))

        # deriving the confusion matrix and reporting the measures
        labels = np.array(labels, dtype=np.uint8).reshape(-1)
        predictions = np.array(predictions, dtype=np.uint8).reshape(-1)
        confusion_matrix = confusion_matrix_from_preds_and_labels(self.reader.num_classes, predictions, labels)
        reporter = ResultReporter(confusion_matrix)
        results_dict = reporter.getResultDict()
        accuracy = results_dict['accuracy'] * 100.0

        # plotting the confusion matrix
        sorted_labels = [*self.reader.class_labels_dict.values()]
        if epoch is not None: 
            test_name = self.args.experiment_name + '_epoch{}'.format(epoch)
        else: 
            test_name = self.args.experiment_name
        conf_folder_path = self.handler.path('confusion_matrices/{}'.format(test_name), use_results_path=True)
        title = '{} - NN Test'.format(test_name)
        reporter.plot_confmat(sorted_labels, title, conf_folder_path, title)

        # Logging
        running_test_loss /= len(self.reader.test_batches)
        self.handler.log('info', 'NN Test{} | Final Result | Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(' - Epoch {}'.format(epoch) if epoch is not None else '', running_test_loss, running_correct, len(self.reader.test_batches)*self.args.batch_size, accuracy))
        
        # writing result to CSV file
        results_list = [running_test_loss, '{:.2f}'.format(accuracy), \
                '{:.4f}'.format(results_dict['logFScoreSum']), '{:.4f}'.format(results_dict['logFScoreMean']), '{:.4f}'.format(results_dict['precisionMacro']), '{:.4f}'.format(results_dict['precisionMicro']), \
                '{:.4f}'.format(results_dict['recallMacro']), '{:.4f}'.format(results_dict['recallMicro']), '{:.4f}'.format(results_dict['fScoreMacro']), '{:.4f}'.format(results_dict['fScoreMicro'])]

        # if the file doesn't exist yet, we first create it and write the column names to the file 
        if not self.handler.check_results_csv_file_path_exists(test_name, subfolder=self.args.results_subfolder, file_name=self.args.nn_results_filename): 
            column_names = ['Timestamp', 'Experiment Name', 'Running Loss', 'Accuracy', 'LogFScoreSum', 'LogFScoreMean', 'PrecisionMacro', 'PrecisionMicro', \
                'RecallMacro', 'RecallMicro', 'fScoreMacro', 'fScoreMicro']
            self.handler.write_result(column_names, test_name, subfolder=self.args.results_subfolder, file_name=self.args.nn_results_filename, writing_column_names=True)
        
        self.handler.write_result(results_list, test_name, subfolder=self.args.results_subfolder, file_name=self.args.nn_results_filename)

        if self.args.tensorboard_logging:
            self.tb.add_scalar('00 General/TestRunningLoss', running_test_loss, 0)
            self.tb.add_scalar('00 General/TestAccuracy', accuracy, 0)

        self.net.train() # set back to training mode as default


    def save_net_for_visualization(self, batch_input, folder_path=None, file_name=None): 
        '''
            Exports the pytorch model to a model in ONNX for better visualization with: https://github.com/lutzroeder/netron 
            Args: 
                batch_input: a batch from the train method for tracing in the network
                folder_path: full folder (including last slash) under which to store the visualization (if None: will be subfolder model_exports within experiment folder)
                file_name: file name (without ending) under which to store the the visualization (if None: will be just experiment name)
        '''

        self.net.eval() # eval mode

        dummy_input = torch.from_numpy(batch_input['features']).float()
        if self.args.cuda: 
            dummy_input = dummy_input.cuda()

        input_names = self.net.input_names
        output_names = self.net.output_names

        if not folder_path: 
            folder_path = self.handler.path('model_exports')

        if not file_name: 
            file_name = self.args.experiment_name

        # ONNX - Model
        torch.onnx.export(self.net, dummy_input, "{}{}.onnx".format(folder_path, file_name), input_names=input_names, output_names=output_names)
        self.handler.log('info', 'Exported ONNX-Model to: {}{}.onnx'.format(folder_path, file_name))

        # PTH - Model
        torch.save(self.net.state_dict(), "{}{}.pth".format(folder_path, file_name))
        self.handler.log('info', 'Exported PTH-Model to: {}{}.pth'.format(folder_path, file_name))
        
        self.net.train() # back to training mode

    
    def write_config_summary(self, args_config_string):
        '''
            Writes and stores a summary of the experiment configurations.
            Args: 
                args_config_string: a string representation of the argparser settings - provided by the ArgsHandler() class
        '''
        assert isinstance(args_config_string, str)
        self.handler.log('debug', 'Writing Summary file for experiment: {}'.format(self.args.experiment_name))

        # write two copies, one in a general summaries folder and one in the actual experiment folder 
        f_list = [self.handler.path('summary') + '{}.txt'.format(self.args.experiment_name), self.handler.path('summaries', use_results_path=True) + '{}.txt'.format(self.args.experiment_name)]
        for filepath in f_list: 
            with open(filepath, "w") as f:

                rep_str = args_config_string

                if not self.args.random_forest_direct: 
                    rep_str += '---- Loss and Optimizer ----\n'
                    rep_str += 'Loss: {}\n'.format(str(self.criterion))
                    rep_str += 'Optimizer: {}\n\n'.format(str(self.optimizer))

                rep_str += '---- Additional DataReader Information ----\n'
                rep_str += 'Number of Training Batches: {}\n'.format(len(self.reader.train_batches))
                rep_str += 'Number of Testing Batches: {}\n'.format(len(self.reader.test_batches))
                if self.args.validation: 
                    rep_str += 'Number of Validation Batches: {}\n\n'.format(len(self.reader.validation_batches))
                else: 
                    rep_str +='\n'

                rep_str += '---- Handler Settings ----\n'
                rep_str += 'Handler Basepath: {}\n'.format(self.handler.basepath)
                rep_str += 'Handler Overwrite: {}\n'.format(self.args.handler_overwrite)
                if not self.args.random_forest_direct: 
                    rep_str += 'Tensorboard Path: {}\n'.format(self.handler.tensorboard_path)

                if not self.args.random_forest_direct: 
                    rep_str += '\n---- Detailed Network Architecture ----\n'
                    rep_str += 'Network Overview:\n\n{}\n\n'.format(self.net)

                f.write(rep_str)

    def save_model(self, subfolder='latest'):
        '''
            Saving the networks weights.
            Args:
                subfolder: The subfolder within the models folder and within the models-folder of the experiment
        '''
        filepath = self.handler.path('models/{}/{}'.format(self.args.experiment_name, subfolder), use_results_path=True) + 'model.pth'
        torch.save(self.net.state_dict(), filepath) # just saves the state dict 
        self.handler.log('debug', 'Saved current model as %s' % filepath)


    def load_model(self, filepath):
        '''
            Loading the networks weights.

            Args:
                filepath: path to the file where the pytorch model of the weights is located 
        '''

        device = torch.device('cpu') # always load on CPU and probably put it back in GPU afterwards       
        self.net.load_state_dict(torch.load(filepath, map_location=device)) # just loads the state dict - i.e. the self.net already has to be the right net
        self.handler.log('info', 'Loaded current model as %s' % filepath)
        return True

    
    def save_snapshot(self, epoch, enforce=False):
        '''
            Creating and saving a snapshot.
            It saves the current parameters and settings of the network and reader once.
            But also calls save_model() to save the network state.
            It potentially overwrites previous savings 'latest' and eventually also 'best'.

            Args:
                epoch: the current epoch
                subfolder: The subfolder within the models folder and within the models-folder of the experiment
                enforce: enforces a snapshot even when the snapshotting interval is currently not met
        '''
        if (epoch+1) % self.args.snapshotting_interval != 0 and not enforce and not epoch == self.args.epochs-1:
            return False

        # in the following we save the training settings only (we do this only once for the best and latest model)
        if self.snapshot_save_parameters: 
            snapshot_dict = {
                'experiment_name' : self.args.experiment_name,
                'net_num_classes' : self.net.num_classes,
                'net_num_features' : self.net.num_features, 
                'net_hidden_out_nodes' : self.net.hidden_out_nodes, 
                'net_layer_connections_dict' : self.net.layer_connections_dict, 
                'net_in_features_dict' : self.net.in_features_dict, 
                'net_binary_weights' : self.args.binary_weights,
                'net_bias' : self.args.bias, 
                'net_hidden_activation' : self.args.hidden_activation,
                'net_dropout' : self.args.dropout, 
                'net_batchnorm' : self.args.batchnorm, 
                'args_datareader' : self.args.datareader, 
                'args_batchsize' : self.args.batch_size, 
                'args_validation_split': self.args.validation_split, 
                'args_test_split' : self.args.test_split,
                'args_data_normalization' : self.args.data_normalization_mode,
                'reader_folderpath' : self.reader.folder_path,
                'reader_num_features' : self.reader.num_features
            }

            if self.args.snapshotting_best:
                self.handler.create_snapshot(snapshot_dict, subfolder='best')
            self.handler.create_snapshot(snapshot_dict, subfolder='latest')

            if self.args.snapshotting_best:
                l = ['latest', 'best']
            else: 
                l = ['latest']
            for sb in l: 
                folder_path = self.handler.path('models/{}/{}'.format(self.args.experiment_name, sb), use_results_path=True)
                np.save(folder_path + 'reader_feature_names', np.array(self.reader.feature_names))
                np.save(folder_path + 'reader_train_features', self.reader.train_features)
                np.save(folder_path + 'reader_train_labels', self.reader.train_labels)
                np.save(folder_path + 'reader_test_features', self.reader.test_features)
                np.save(folder_path + 'reader_test_labels', self.reader.test_labels)
                np.save(folder_path + 'reader_validation_features', self.reader.validation_features)
                np.save(folder_path + 'reader_validation_labels', self.reader.validation_labels)

            self.snapshot_save_parameters = False # to not store it again in the next round 

        # save best and last model (we do this everytime we snapshot)
        saved_best = False
        
        if self.args.snapshotting_best and self.expfct(self.val_scores) == self.val_scores[-1]:
            self.save_model(subfolder='best')
            saved_best = True

        # save latest model
        self.save_model(subfolder='latest')
        return saved_best


    def load_snapshot(self, folderpath):
        '''
            Loading a snapshot - includes the network weights and the parameter settings of the reader and the network.

            Args:
                folderpath: path to the folder where the snapshotted files and the model files are located (including last slash)
            Returns:
                True if loading was successful, False if loading was not possible 
        '''
       
        # loading latest model
        snapshot_dict = self.handler.load_snapshot(filepath=folderpath+'snapshot.json')

        if self.args.snapshotting and not self.args.snapshotting_overwrite: 
            error_msg = 'Loading snapshot cannot be fulfilled because you would overwrite an exisiting snapshot in the next epcosh with the same experiment name. If you wish to overwrite, call --snapshotting_overwrite or change the experiment name!'
            assert self.handler.path('models/{}'.format(self.args.experiment_name), use_results_path=True) not in folderpath, error_msg
            inform_string = 'binary_weights, bias, hidden_activation, dropout, batchnorm, batchsize, validation_split, test_split, data_normalization, datareader'
            self.handler.log('warning', 'Loading snapshot, the following argparser arguments will be ignored: {}. Settings from the network and data reader will be called from the snapshot.'.format(inform_string))

        if snapshot_dict is not None:
            net_num_classes = snapshot_dict['net_num_classes']
            net_num_features = snapshot_dict['net_num_features']
            net_hidden_out_nodes = snapshot_dict['net_hidden_out_nodes']
            self.args.hidden_layer_outnodes = net_hidden_out_nodes

            pre_net_layer_connections_dict = snapshot_dict['net_layer_connections_dict']
            # NOTE there is a problem with loading from json file that the int-keys from the dictionary were replaced by string-keys, i.e. need of changing it back
            net_layer_connections_dict = {}
            for k, v in pre_net_layer_connections_dict.items(): 
                net_layer_connections_dict[int(k)] = v

            pre_net_in_features_dict = snapshot_dict['net_in_features_dict']
            # NOTE there is a problem with loading from json file that the int-keys from the dictionary were replaced by string-keys, i.e. need of changing it back
            net_in_features_dict = {}
            for k, v in pre_net_in_features_dict.items(): 
                net_in_features_dict[int(k)] = v

            self.args.binary_weights = snapshot_dict['net_binary_weights']
            self.args.bias = snapshot_dict['net_bias']
            self.args.hidden_activation = snapshot_dict['net_hidden_activation']
            self.args.dropout = snapshot_dict['net_dropout']
            self.args.batchnorm = snapshot_dict['net_batchnorm']
            self.args.datareader = snapshot_dict['args_datareader']
            self.args.batch_size = snapshot_dict['args_batchsize']
            self.args.validation_split = snapshot_dict['args_validation_split']
            self.args.test_split = snapshot_dict['args_test_split']
            self.args.data_normalization_mode = snapshot_dict['args_data_normalization']
            
            if str.lower(self.args.datareader) == 'coimbra': 
                self.reader = CoimbraReader(self.args, load_from_snapshot=True)

            else: # meaning MNIST
                self.reader = MNISTReader(self.args, load_from_snapshot=True)

            self.reader.num_features = snapshot_dict['reader_num_features']
            self.reader.folder_path = snapshot_dict['reader_folderpath']
            self.reader.train_features = np.load(folderpath + 'reader_train_features.npy')
            self.reader.train_labels = np.load(folderpath + 'reader_train_labels.npy')
            self.reader.test_features = np.load(folderpath + 'reader_test_features.npy')
            self.reader.test_labels = np.load(folderpath + 'reader_test_labels.npy')
            self.reader.validation_features = np.load(folderpath + 'reader_validation_features.npy')
            self.reader.validation_labels = np.load(folderpath + 'reader_validation_labels.npy')
            self.reader.feature_names = np.load(folderpath + 'reader_feature_names.npy').tolist()

            self.reader.shuffle_and_tie_new_batches()
            self.reader.shuffle_and_tie_new_batches(test_data=True)

            self.logic_processor.args = self.args 

            # create the network before loading the weights 
            self.net = LogicNN(self.args, num_features=net_num_features, num_classes=net_num_classes, hidden_layer_outnodes=net_hidden_out_nodes, layer_connections_dict=net_layer_connections_dict, in_features_dict=net_in_features_dict)
            self.load_model(folderpath+'model.pth')
            if self.args.cuda: 
                self.net.cuda()
            else: 
                self.net.cpu()
            return True 
        else: 
            return False 
