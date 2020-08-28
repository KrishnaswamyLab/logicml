import numpy as np
import os
import pathlib
from logicml.code.pipeline.handler import * 
from logicml.code.pipeline.utils import * 

class LogicSimulator: 

    def __init__(self, handler, args, data_features, data_labels=None, folder_path=None, results_folder_path=None): 
        '''
            This class provides all functionalities for simulating the logic of Verilog-files. 
            It can be used for restoring logic snapshots and simulating the logic independently from its creation time point.
            It also provide functionalities to either simulate just single logic modules or the overall full logic. 
            It can also report outputs of intermediate signals.
            
            Args:  
                handler: object of the Handler class, used for logging and other utilities
                args: the argparser argument (provided by the ArgsHandler-class)
                data_features: np array of features of the test data observations with which to simulate the logic (can be None, if only wanting to derive confusion matrix from simulation text-file)
                data_labels: np array of labels of the test data observations with which to simulate / evaluate the logic (optional)
                folder_path: [optional] full path to the folder in which to store the derived files and simulation results (if None: a simulation/experiment_name subfolder will be created in the handler's result folder)
                results_folder_path: optional path to the top folder where the experiment folders are located (if None, it will be searched for experiment folder within the handler's results folder)
        '''
        
        # NOTE: a  typical workflow is the following: 
        #   - create an instance of the LogicSimulator with the right data handler and settings with which you want to simulate 
        #   - add multiple logic experiments (potentially also with intermediate outputs and optimization commands)
        #   - run the simulation of all previously added logic experiments at once (maybe also with debugging options) 

        assert isinstance(handler, Handler)
        self.handler = handler

        self.args = args
        self.data_features = data_features

        if data_labels is not None: 
            data_labels = data_labels.reshape(-1)
            if self.data_features is not None: 
                assert self.data_features.shape[0] == data_labels.shape[0], 'LogicSimulator: Provided data_features array has {} observations, but {} labels provided.'.format(self.data_features.shape[0], data_labels.shape[0])
        
        self.data_labels = data_labels

        self.results_folder_path = results_folder_path
        
        # if the path to the folder is provided, but the last slash is missing - add it 
        if self.results_folder_path is not None and self.results_folder_path[-1] != '/': 
            self.results_folder_path += '/'

        # this dictionary will store detailed information about all the logic to simulate 
        # it will be filled within load_logic_from_previous_experiment()
        self.eqn_files_dict = {}


        self.experiment_folder = None # will be filled if load_logic_from_previous_experiment() is called

        if folder_path is None: 
            self.folder_path = self.handler.path('simulations/{}'.format(self.args.experiment_name), use_results_path=True)
        else: 
            if not folder_path.endswith('/'):
                folder_path += '/'
            self.folder_path = folder_path
        

    def load_test_data_snapshot(self, folder_path): 
        '''
            This method loads the test features numpy array and labels array that was created by the Trainer class for future simulations. 
            This means it also resets self.data_features and self.data_labels.
            NOTE: therefore this method should potentially be called before calling the simulate()-method. 
            NOTE: this method assumes that there is a features.npy and a labels.npy file in folder_path.
            NOTE: when wanting to call this method, you can also initialize the LogicSimulator object's data with just None
            Args: 
                folder_path: the full path to the folder where the data is located (including last slash)
        '''

        assert isinstance(folder_path, str), 'LogicSimulator: Loading test data snapshot failed. Given folderpath is not a string.'
        if not folder_path.endswith('/'):
            folder_path += '/'
        fp = folder_path + 'features.npy'
        lp = folder_path + 'labels.npy'
        assert os.path.isfile(fp), 'LogicSimulator: Loading test data snapshot failed. File {} does not exist.'.format(fp)
        assert os.path.isfile(lp), 'LogicSimulator: Loading test data snapshot failed. File {} does not exist.'.format(fp)
        
        labels = np.load(lp)
        features = np.load(fp)
        assert features.shape[0] == labels.shape[0], 'LogicSimulator: Loading test data snapshot failed. Provided data_features array has {} observations, but {} labels provided.'.format(features.shape[0], labels.shape[0])
        self.data_features = features 
        self.data_labels = labels
        self.handler.log('info', 'LogicSimulator: Loaded test data snapshot from: {}'.format(folder_path))
    
    def add_logic_experiment(self, logic_file_path, intermediate_outputs=None, keep_intermediate_files=False, only_intermediate_outputs=False, optimization_command=None): 
        '''
            This method can load a logic file (either Verilog or eqn-file) and for simulation with the data. 
            It can also take intermediate signal names and will create an extra simulation for those signals (only supported for Verilog-files and signal name must match a declaration in the Verilog file).
            NOTE: this method can be called multiple times to add multiple experiments to simulate - when calling the simulate() method, all of the stored logic simulations will be run
            Args: 
                logic_file_path: the full path to the logic file (can either be .v-file or .eqn-file)
                intermediate_outputs: 
                keep_intermediate_files: Boolean to set to True if eqn-files and Verilog-files that are created at intermediate stages should be kept for the reason of inspection and debugging
                only_intermediate_outputs: Boolean to set to True if for the case of providing intermediate_outputs, the final output of the logic should not be simulated
                optimization_command: command for additional optimization on AIG - NOTE: only applied to full Verilog logic (ignored if you load a eqn-file and also ignored for deriving intermediate signals)
        '''

        # TODO: need to add a file_name option under which the simulation results of the added logic experiment are created? 

        if optimization_command is not None: 
            assert optimization_command in ['dc2', 'mfs_advanced', 'syn2'], 'LogicSimulator: Optimization Command {} was used that is not declared. Please use None or one of following: {}'.format(optimization_command, ['dc2', 'mfs_advanced', 'syn2'])
        assert isinstance(logic_file_path, str)
        assert logic_file_path.endswith('.v') or logic_file_path.endswith('.eqn'), 'LogicSimulator: Cannot load logic - only eqn- and Verilog-files supported.'
        
        if logic_file_path.endswith('.eqn'): 
            assert intermediate_outputs is None, 'LogicSimulator: Cannot load logic - for eqn-files, intermediate outputs are not supported.'
            d = self._get_input_output_information_from_eqn_file(logic_file_path)
            
            # NOTE: for given eqn-file the quantization scheme is not given in a top comment before the Verilog module declaration
            # therefore we need to at least assert that the number of total bits provided by the args fits to the declarations in the eqn-file
            error_msg = 'LogicSimulator: The total bits number [{}] of the quantization scheme with which you try load the file {} does not fit to the declaration of input bit width [{}] in the file'.format(self.args.total_bits, logic_file_path, d['in_bit_num'])
            assert d['in_bit_num'] == self.args.total_bits, error_msg
            
            module_name = logic_file_path.split('.eqn')[0].split('/')[-1]
            d['total_bits'] = self.args.total_bits
            d['fractional_bits'] = self.args.fractional_bits
            d['delete'] = False
            d['filepath'] = logic_file_path
            self.eqn_files_dict[module_name] = d
            self.handler.log('info', 'LogicSimulator - Prepared logic for simulation: {}'.format(logic_file_path))

        else: # meaning it's a Verilog file 
            module_name = logic_file_path.split('.v')[0].split('/')[-1]
            if optimization_command is not None: 
                module_name += '_optim_{}'.format(optimization_command)

             # creating the eqn file for the full logic simulation
            if (intermediate_outputs is None) or (intermediate_outputs is not None and not only_intermediate_outputs): 
                total_bits, fractional_bits = self._get_quantization_scheme_declarations_from_verilog_module(logic_file_path)
                eqn_filepath = self._verilog_to_eqn_file(logic_file_path, module_name, optimization_command) # NOTE: support of optimization command
                d = self._get_input_output_information_from_eqn_file(eqn_filepath)
                d['filepath'] = eqn_filepath
                d['total_bits'] = total_bits
                d['fractional_bits'] = fractional_bits
                if keep_intermediate_files: 
                    d['delete'] = False
                else: 
                    d['delete'] = True
                self.eqn_files_dict[module_name] = d
                self.handler.log('info', 'LogicSimulator - Prepared logic for simulation: {}'.format(eqn_filepath))

             # creating the eqn files for the simulations of intermediate output signals  
            if intermediate_outputs is not None:
                assert isinstance(intermediate_outputs, list), 'LogicSimulator.add_logic_experiment(): the provided intermediate_outputs must be a list of string identifiers.'
                for intermediate_out in intermediate_outputs: 
                    assert isinstance(intermediate_out, str), 'LogicSimulator.add_logic_experiment(): elements inside the intermediate_outputs list must be string identifiers.'
                    new_module_name = module_name+'_'+intermediate_out
                    v_filepath = self._create_verilog_intermediate_output(logic_file_path, intermediate_out, new_module_name)
                    total_bits, fractional_bits = self._get_quantization_scheme_declarations_from_verilog_module(v_filepath)
                    eqn_filepath = self._verilog_to_eqn_file(v_filepath, new_module_name, None) # NOTE: no optimization supported for intermediate signals

                    if not keep_intermediate_files: # delete intemediate Verilog file
                        os.system('rm {}'.format(v_filepath))

                    d = self._get_input_output_information_from_eqn_file(eqn_filepath)
                    d['filepath'] = eqn_filepath
                    d['total_bits'] = total_bits # self.args.total_bits
                    d['fractional_bits'] = fractional_bits # self.args.fractional_bits
                    if keep_intermediate_files: 
                        d['delete'] = False
                    else: 
                        d['delete'] = True
                    self.eqn_files_dict[new_module_name] = d
                    self.handler.log('info', 'LogicSimulator - Prepared logic for simulation: {}'.format(eqn_filepath))


    def simulate(self, debug=False, include_features_debug=True, num_observations_to_run=None): 
        '''
            This method executes the simulation of all logic experiments that have been loaded with the method add_logic_experiment(). 
            Args: 
                debug: Boolean to set to True if additional information should be printed to the simulation results file (for the reason of debugging)
                include_features_debug: Boolean to set to True if in case of debugging mode, also each feature binarization should be written to file
                num_observations_to_run: [optional] integer number that state how many of the observations in the test data should be evaluated (can e.g. be useful for debugging with just a few examples)
            Returns: 
                a list with all full paths to the create simulation results text-files
        '''
        
        # this list will store the filepaths to all simulation result text-files (e.g. for calling confmat creation afterwards)
        results_fp_list = []

        assert len(self.eqn_files_dict.keys()) > 0, 'LogicSimulator: Call of method simulate() failed because no logic experiments were loaded. First call add_logic_experiment().'
        for k, v in self.eqn_files_dict.items(): 
            results_fp = self._single_simulation(v, k, debug=debug, include_features_debug=include_features_debug, num_observations_to_run=num_observations_to_run)
            results_fp_list.append(results_fp)
            if v['delete']: 
                os.system('rm {}'.format(v['filepath']))
        return results_fp_list


    def get_possible_logic_simulation_files(self, experiment_name, print_out=True): 
        '''
            This method searches in the results folder for the experiment_name folder and then searches in all subdirectories for Verilog and eqn files.
            This can help when searching for specific logic modules or final logic that you want to simulate or derive something from.
            Args: 
                experiment_name: the experiment_name in the results folder you are interested in 
                print_out: Boolean to set to True if the paths to the files should be printed 
            Returns: 
                a list of all eqn-files and Verilog files in the experiment_name folder 
        '''

        # get all the subfolders names within the results folder
        experiment_folders = self._get_experiment_folders()

        # check if there is a subfolder that matches this names - must be the case, otherwise cancelled 
        if experiment_name not in experiment_folders: 
            if print_out: 
                # this is no assert to not quit the whole program, becuase this method is not important enough to do that 
                print('LogicSimulator - Cannot load logic with experiment name [{}] because it does not exist at path: {}.'.format(experiment_name, self.results_folder_path)) 
            return None
        
        # if a match was found, build the path to that folder
        self.experiment_folder = self.results_folder_path + experiment_name

        # get a list of paths to all files in the directory
        file_paths = []
        for (dirpath, dirnames, filenames) in os.walk(self.experiment_folder):
            file_paths += [os.path.join(dirpath, file) for file in filenames]

        # now get a list of all eqn and Verilog files in that folder
        verilog_files = [] 
        eqn_files = [] 

        for fp in file_paths: 
            if fp.endswith('.v'):
                verilog_files.append(fp)
            elif fp.endswith('.eqn'):
                eqn_files.append(fp)

        if print_out: 
            print('------------------')
            print('Verilog Files: ')
            print('------------------\n')
            for fp in verilog_files: 
                print(fp)
            print('\n------------------')
            print('EQN Files: ')
            print('------------------\n')
            for fp in eqn_files: 
                print(fp)
            print('\n')

        return verilog_files.extend(eqn_files)


    def calculate_confmat_from_simulation_results(self, result_file_path, num_classes):
        '''
            Method that takes the simulation results text file and calculates the confusion matrix from it.
            NOTE: This method only works if:  
                - the labels were also written to the simulation results file 
                - or the class was initialized with data_labels not being None
            Args: 
                result_file_path: path to the .txt-file in which the simulation results where written 
                num_classes: the number of classes
            Returns: 
                numpy array of confusion matrix (rows = predictions, columns = true labels)
        '''

        # NOTE: in the following, only the information about the simulation results is extracted from the textfile
        # NOTE: this means that it is not proven that the labels actually match to the simulation results - this is up to the user
        self.handler.log('info', 'LogicSimulator: Creating confusion matrix from results: {}'.format(result_file_path))

        sim_results_array, test_labels_array = self._get_sim_results_and_labels(result_file_path)
        assert test_labels_array is not None, 'LogicSimulator: Cannot create confusion matrix from simulation results, because labels were not written to the simulation file and also no labels are externally provided.'
        max_sim = int(np.max(sim_results_array))
        assert max_sim < num_classes, 'LogicSimulator: Cannot create confusion matrix - maximum class prediction in simulation is {}, but number of classes declared is {}.'.format(max_sim, num_classes)
        return confusion_matrix_from_preds_and_labels(num_classes, sim_results_array, test_labels_array)


    def save_simulation_results_array(self, result_file_path, file_name, folder_path=None):
        '''
            Method that takes the simulation results text file, calculates the simulation results array from it and stores it as numpy file.
            Args: 
                result_file_path: path to the .txt-file in which the simulation results where written 
                file_name: file name under which to store the numpy array of the simulation results (without ending)
                folder_path: [optional] the full path to the folder in which to store the numpy array of the simulation results (including last slash), if None the internal path representation will be chosen
        '''
        
        if folder_path is not None: 
            assert isinstance(folder_path, str), 'LogicSimulator: Saving simulation results array failed. Given folderpath is not a string.'
            assert os.path.isdir(folder_path), 'LogicSimulator: Saving simulation results array failed, because the given folder does not exist: {}'.format(folder_path)
            if not folder_path.endswith('/'):
                folder_path += '/'
            fp = folder_path + file_name
        else: 
            fp = self.folder_path + file_name

        sim_results_array, test_labels_array = self._get_sim_results_and_labels(result_file_path)
        np.save(fp, sim_results_array)
        self.handler.log('info', 'LogicSimulator: Saved simulation results array: {}'.format(fp+'.npy'))


    def _get_sim_results_and_labels(self, result_file_path): 
        '''
            This method reads a simulation txt file and extract the simulation results. 
            If the solutions/labels were also written to that file, it extracts it, otherwise, the self.data_labels declaration is used.
            Args: 
                result_file_path: path to the .txt-file in which the simulation results where written 
            Returns: 
                sequence of simulation results numpy array and labels array (the latter can also be None in certain cases)
        '''
        # TODO: additional assertions? 
        # list to store the simulation results in --> will be turned into np array
        sim_results = []

        # list to store the labels if they were written to the simulation file
        sim_class_labels = []

        # reading results from file 
        with open(result_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines: 
                if 'Simulation' in line: # other lines might refer to lines that were written in debugging mode
                    
                    if 'Class Label' in line: # if the simulation was done including a report of label result
                        sim_res = int(line.split(': ')[-2].split(' (')[0])
                        class_label = int(line.split(': ')[-1].split(')')[0])
                        sim_class_labels.append(class_label)
                    else: # if the simulation was done without labels in the simulation report
                        sim_res = line.split(': ')[-1]
                        sim_res = int(sim_res.split('\n')[0])
                    sim_results.append(sim_res)

        # preparing simulation results and intended outcomes
        sim_results = np.array(sim_results, dtype=np.uint8)

        if len(sim_class_labels) == 0: 
            if self.data_labels is not None: 
                test_labels_array = np.array(self.data_labels, dtype=np.uint8).reshape(-1)
                assert test_labels_array.shape[0] == sim_results.shape[0], 'LogicSimulator: number of observations in the test labels array [{}] does not match the shape of the simulation results [{}].'.format(test_labels_array.shape[0], sim_results.shape[0])
            else: 
                test_labels_array = None
        else: 
            test_labels_array = np.array(sim_class_labels, dtype=np.uint8).reshape(-1)
            assert test_labels_array.shape[0] == sim_results.shape[0], 'LogicSimulator: number of observations in the test labels array [{}] does not match the shape of the simulation results [{}].'.format(test_labels_array.shape[0], sim_results.shape[0])
        
        return sim_results, test_labels_array


    def _get_input_output_information_from_eqn_file(self, eqn_filepath): 
        '''
            This method extracts information from the eqn-file about:
            - number of inputs 
            - bitwidth of the inputs
            - input identifier
            - bitwidth of the outputs
            - output identifier

            Args: 
                eqn_filepath: full path to the eqn file
            Returns: 
                above mentioned information in a dictionary
        '''

        # NOTE: some assumptions are made here that all apply to be True in this developed framework: 
        # - the identifiers of the inputs (i.e. the input names) are all same, except for the count number being different
        # - the inputs are assumed to all have the same bit width 
        # - there is just one output per logic module (with probably varying output bit size)
        # - the first input always has the number 0 --> e.g. final_in0
        # - the inputs are numbered without wholes, e.g. final_in0, final_in2 but no existence of final_in1 is not allowed
        # - the input declaration starts with the key INORDER
        # - the output declaration starts with the key OUTORDER

        input_identifier = ''
        output_identifier = ''
        inputs_number = 0
        input_bit_number = 0
        output_bit_number = 0

        with open(eqn_filepath, 'r') as f: 
            lines = f.readlines()
            inorder_flag = False # a flag to mark that the input declaration starts
            outorder_flag = False # a flag to mark that the input declaration has finished
            for line in lines: 
                
                # -------- End of Input and Output Declarations ------------
                if not ('ORDER' in line) and ('=' in line) and outorder_flag: 
                    break

                # -------- Handling Input Declarations ------------
                if 'INORDER' in line: # first line of input declarations
                    splits = line.split('INORDER = ')[-1].split('[') 
                    # setting input identifier once and flag that input declaration started
                    inorder_flag = True
                    splits_new = splits[0].split('0')
                    for i in range(len(splits_new)-1): 
                        s = splits_new[i]
                        input_identifier += s
                        if i != len(splits_new)-2:
                            input_identifier += '0'

                if inorder_flag and not outorder_flag and not 'INORDER' in line: # still input declaration, but not first line
                    splits = line.split('[') 

                if inorder_flag and not outorder_flag: # processing
                    for s in splits: 
                        if input_identifier in s: # updating the number of inputs
                            i = int(s.split(input_identifier)[-1]) 
                            if i > inputs_number: 
                                inputs_number = i
                        if ']' in s: # updating the number of input bits
                            i = int(s.split(']')[0]) 
                            if i > input_bit_number: 
                                input_bit_number = i

                # -------- Handling Output Declarations ------------
                if 'OUTORDER' in line: # first line of output declarations
                    splits = line.split('OUTORDER = ')[-1].split('[') 
                    # setting input identifier once and flag that input declaration started
                    outorder_flag = True
                    output_identifier = splits[0]
                
                if outorder_flag and not 'OUTORDER' in line: # still output declaration, but not first line
                    splits = line.split('[') 

                if outorder_flag: # processing
                    for s in splits: 
                        if ']' in s: # updating the number of input bits
                            i = int(s.split(']')[0]) 
                            if i > output_bit_number: 
                                output_bit_number = i

        # -------- Summarizing Collected Information ------------
        # we start everything with index 0, so the number of elements must be increased by 1
        inputs_number += 1 
        input_bit_number += 1
        output_bit_number += 1

        d = {
            'in_num' : inputs_number, 
            'in_bit_num' : input_bit_number, 
            'in_id' : input_identifier, 
            'out_bit_num' : output_bit_number,
            'out_id' : output_identifier
            }

        return d


    def _get_quantization_scheme_declarations_from_verilog_module(self, logic_file_path): 
        '''
            This method extracts the information about the quantization scheme from the top line of the Verilog-file. 
            NOTE: This is an assumption of how the Verilog file is always created in the same manner in this pipeline.
            NOTE: It serves the reason to simulate Verilog logic correctly even if the exact quantization scheme is not known anymore and not correctly defined with the argparser arguments.
            NOTE: This is actually an additional feature that is provided for simulation of Verilog files, but is something that can not be assured for the eqn-files.
            Args: 
                logic_file_path: the full path to the Verilog file
            Returns: 
                sequence of total bits (int) and fractional bits (int)
        '''

        assert isinstance(logic_file_path, str)
        assert logic_file_path.endswith('.v'), 'LogicSimulator: Can only extract information about the quantization scheme from Verilog files.'
        total_bits = None
        fractional_bits = None
        with open(logic_file_path, 'r') as f: 
            lines = f.readlines()
            for line in lines: 
                if 'Quantization Scheme' in line: 
                    total_bits = int(line.split('Total Bits: ')[-1].split(' -')[0])
                    fractional_bits = int(line.split('Fractional Bits: ')[-1].split('\n')[0])
        assert (total_bits is not None) and (fractional_bits is not None), 'LogicSimulator: Unable to find correct quantization scheme declaration in: {}'.format(logic_file_path)
        return total_bits, fractional_bits


    def _create_verilog_intermediate_output(self, logic_file_path, intermediate_out, file_name): 
        '''
            This method takes a Verilog file and searches for the signal where the intermediate output identifier is assigned. 
            If this signal exists, it creates a new Verilog file that encodes exactly the calculations up to this point and makes the intermediate signal the output of the modified Verilog module. 
            Args: 
                logic_file_path: full path to the Verilog file 
                intermediate_out: string identifier of the wire/signal that should be outputted 
                file_name: name for storing the new Verilog file (without .v extension)
            Returns: 
                the filepath to the created Verilog-file
        '''
        new_verilog_file = self.folder_path + file_name + '.v'

        with open(logic_file_path, 'r') as f: 
            lines = f.readlines()
            dec_signal_line_number = None # variable that will store at which line number the intermediate signal is declared
            assign_signal_line_number = None # variable that will store at which line number the intermediate signal is assigned
            output_line_number = None # variable that will store at which line number the output wire declaration takes place
            
            # start searching for the indices in the lines list, where information has to be replaced or is or relevant informations ends 
            for idx, line in enumerate(lines): 
                if 'assign {} = '.format(intermediate_out) in line: 
                    assign_signal_line_number = idx 
                if ('wire' in line) and (intermediate_out in line) and not ('=' in line): 
                    dec_signal_line_number = idx
                if ('output' in line) and not ('//' in line):
                     output_line_number = idx
                if (dec_signal_line_number is not None) and (assign_signal_line_number is not None) and (output_line_number is not None): # means we found everything we wanted
                    break
            
            assert output_line_number is not None, 'LogicSimulator: failed to find the output declaration of the Verilog file: {}'.format(logic_file_path)
            assert dec_signal_line_number is not None, 'LogicSimulator: failed to find the declaration of the intermediate signal {} in the Verilog file: {}'.format(intermediate_out, logic_file_path)
            assert dec_signal_line_number is not None, 'LogicSimulator: failed to find the assignment of the intermediate signal {} in the Verilog file: {}'.format(intermediate_out, logic_file_path)

            with open(new_verilog_file, 'w') as f_new:
                # replacing the output declaration of the Verilog module
                s = '    output ' + lines[dec_signal_line_number].replace('    ', '')
                s = s.replace(';', '')
                lines[output_line_number] = s

                # delete the original declaration of the intermediate signal
                del lines[dec_signal_line_number]

                # writing the lines to the file up to the point where assignment of the intermediate output happens
                # NOTE: the declaration of the signal in the Verilog file always happened before the assignment, thus deletion of declaration line affects assignment index
                # NOTE: one line was deleted - therefore, assign_signal_line_number can now be used as index to mark the end (with being excluded)
                for line in lines[:assign_signal_line_number]: 
                    f_new.write(line)
                f_new.write('endmodule')
        return new_verilog_file


    def _get_abc_execution_path(self): 
        '''
            This method is a helper method to get the path to the ABC executable.
            NOTE: It assumes the Git-Structure that the ABC-executable is located one folder above this script and then in a subfolder called abc_synthesis.
            Returns: 
                path to the ABC executable
        '''
        # get absolute path to the abc executable
        # first get the directory of this python file
        abc_exec_path = os.path.dirname(os.path.abspath(__file__))
        # now, go up one folder in this git and go to abc_synthesis folder
        abc_exec_path = os.path.dirname(abc_exec_path) + '/abc_synthesis/abc'
        return abc_exec_path


    def _verilog_to_eqn_file(self, logic_file_path, file_name, optimization_command): 
        '''
            This method takes a path to a Verilog-file and derives an eqn-file from that 
            Args: 
                logic_file_path: the path to the Verilog file
                file_name: file_name of the created eqn file (without .eqn ending)
                optimization_command: additional AIG optimization command - can be a keyword or None for no optimization
            Returns: 
                the full path to the created eqn file
        '''
        assert isinstance(logic_file_path, str)
        assert logic_file_path.endswith('.v'), 'LogicSimulator: Translation to eqn-file only supports Verilog-file as input.'
        abc_exec_path = self._get_abc_execution_path()

        # creating the shell script file
        script_filepath = self.folder_path + '{}_eqn_creation.sh'.format(file_name)
        with open(script_filepath, "w") as f:
            # opening up the interactive abc command line tool
            f.write('{} << COMMANDS\n'.format(abc_exec_path))
            # adding the commands that should be executed in there
            f.write('%read {}\n'.format(logic_file_path))
            f.write('%blast\n')
            f.write('&put\n')  
            
            # handling potential optimization
            if optimization_command == 'dc2':
                # less scalable but is guaranteed to never increase the size
                if str.lower(self.args.verbosity) == 'debug':
                    optim_command = '&dc2 -v'
                else: 
                    optim_command = '&dc2'

            elif optimization_command  == 'mfs_advanced':
                # switches to restrict the depth of nodes before performing a don't care minimization 
                if str.lower(self.args.verbosity) == 'debug':
                    optim_command = '&syn2 -w\n&if -K 6\n&mfs -w\n&st'
                else: 
                    optim_command = '&syn2\n&if -K 6\n&mfs\n&st'
            
            elif optimization_command == 'syn2': 
                # &syn2 = most powerful exact synthesis / optimization (performs unmapping and mapping) 
                # finds shared nodes, etc. but keeps original logic function - works well up to 10 mio. nodes, but may increase size
                if str.lower(self.args.verbosity) == 'debug':
                    optim_command = '&syn2 -w'
                else: 
                    optim_command = '&syn2'

            if optimization_command is not None:    
                f.write('{}\n'.format(optim_command))
                       
            # exporting the eqn file and overwriting the logic_file_path
            logic_file_path = self.folder_path + '{}.eqn'.format(file_name)
            f.write('write_eqn {}\n'.format(logic_file_path))
            f.write('quit\n')
            f.write('COMMANDS\n')

        # executing the created shell script
        os.system('source {}'.format(script_filepath))
        # delete the shell script after is was executed
        os.system('rm {}'.format(script_filepath))
        return logic_file_path


    def _get_experiment_folders(self): 
        '''
            This method searches through the results folder of the handler object and extracts the experiment names from the first level subfolders. 
            Returns: 
                list of subfolder names of the results_folder_path (if not None) or the handler's results_folder_path
        '''
        if self.results_folder_path is None: 
            self.results_folder_path = self.handler.results_path

        return [f.name for f in os.scandir(self.results_folder_path) if f.is_dir()]


    def _single_simulation(self, eqn_simulation_dict, file_name, debug=False, include_features_debug=True, num_observations_to_run=None):
        '''
            Method that writes a testbench and simulates the final logic with Python based on traversal of the AIG that comes from ABC.
            Args: 
                eqn_simulation_dict: a dictionary, which is an element in self.eqn_files_dict and stores all the needed information.
                file_name: name that should be given to the file that stores the simulation results (without any ending)
                debug: Boolean to set to True if additional information should be printed to the simulation results file (for the reason of debugging)
                include_features_debug: Boolean to set to True if in case of debugging mode, also each feature binarization should be written to file
                num_observations_to_run: [optional] integer number that state how many of the observations in the test data should be evaluated (can e.g. be useful for debugging with just a few examples)
            Returns: 
                full path to the text file with the simulation results 
        '''

        assert type(self.data_features).__module__ == 'numpy', 'LogicSimulator: Data Features must be a matching numpy array to simulate the logic.'
        assert self.data_features.shape[1] == eqn_simulation_dict['in_num'], 'LogicSimulator: The provided data has {} features and does not match the the logic module declaration which takes {} inputs. Simulation failed.'.format(self.data_features.shape[1], eqn_simulation_dict['in_num'])
        # keys in eqn_simulation_dict: 'filepath', 'total_bits', 'fractional_bits', 'delete', 'in_num', 'in_bit_num', 'in_id', 'out_bit_num', 'out_id'

        if num_observations_to_run is not None: 
            assert isinstance(num_observations_to_run, int), 'LogicSimulator: Simulation failed. Parameter num_observations_to_run needs to be integer or None, but was given {}'.format(type(num_observations_to_run))
            if num_observations_to_run > self.data_features.shape[0]: # clipping, just use the maximum number available
                num_observations_to_run = self.data_features.shape[0]
        else: 
            num_observations_to_run = self.data_features.shape[0]

        self.handler.log('info', 'LogicSimulator - Starting Simulation of {}'.format(file_name))

        # get a list with the name of outputs 
        output_names_list = []
        # NOTE: we have always just one output but with multiple bits 
        for b in range(eqn_simulation_dict['out_bit_num']): 
            output_names_list.append('{}[{}]'.format(eqn_simulation_dict['out_id'], b))

        file_path = self.folder_path + file_name +'_sim_results.txt'
        with open(file_path, 'w') as f:  
            constant_output_flag = True

            # -------- Iteration over observations -------------
            for i in range(num_observations_to_run): # iterating through all test data observations 

                self.intermediate_results_dict = {} # this dict will be filled during runtime 

                # -------- Iteration over the features -------------
                for j in range(self.data_features.shape[1]):
                    int_repr = convert_float_to_quantized_int_repr(float(self.data_features[i][j]), eqn_simulation_dict['fractional_bits'], eqn_simulation_dict['total_bits'])
                    bin_str = convert_quantized_int_repr_to_binary_string(int_repr, eqn_simulation_dict['total_bits'])
                    
                    # -------- Assigning the inputs -------------
                    for b in range(eqn_simulation_dict['total_bits']): 
                        # NOTE: we need to adapt to the binary notation and reverse the bitstring: LSB = bit 0 needs to stand at the beginning of the string for accessing the sting the Pythonic way
                        self.intermediate_results_dict['{}{}[{}]'.format(eqn_simulation_dict['in_id'], j, b)] = bool(int(bin_str[::-1][b]))

                # -------- Starting actual simulation -------------
                with open(eqn_simulation_dict['filepath'], 'r') as other_f: 
                    lines = other_f.readlines()
                    for line in lines: 
                        # just evaluaations of lines where compuations of equations take place 
                        if ('=' in line) and (('*' in line) or ('+' in line)): 
                            constant_output_flag = False
                            out_name = line.split(' =')[0]
                            
                            # handling the actual equation in the following
                            self.intermediate_results_dict[out_name] = self._evaluate_eqn_line(line)
                        
                        elif ('=' in line) and not ('ORDER' in line): # means constant output 
                            out_name = line.split(' =')[0]
                            c_out = line.split('= ')[-1].split(';')[0]
                            if '!' in c_out: 
                                negate = True 
                                c_out = c_out.replace('!', '')
                            else: 
                                negate = False
                            if ('new' in c_out) or ('in' in c_out) or ('out' in c_out) or ('[' in c_out): # means constant output, but dependend on other ouput
                                c_out = self.intermediate_results_dict[c_out]
                                if negate: 
                                    c_out = not c_out
                            self.intermediate_results_dict[out_name] = bool(int(c_out))

                final_binary_result_string = ''
                for out in reversed(output_names_list): 
                    final_binary_result_string += str(int(self.intermediate_results_dict[out]))
                
                final_class_predict = convert_binary_string_to_quantized_int_repr(final_binary_result_string) # NOTE: this is the class prediction already as integer number 

                if self.data_labels is not None: # labels provided and also written to simulation result
                    s = 'Simulation Run {} - {}: {} (Class Label: {})\n'.format(i, eqn_simulation_dict['out_id'], final_class_predict, int(self.data_labels[i]))
                else: 
                    s = 'Simulation Run {} - {}: {}\n'.format(i, eqn_simulation_dict['out_id'], final_class_predict)
                
                if debug: # writing addtional information to the file 
                    if include_features_debug:
                        for j in range(self.data_features.shape[1]):
                            int_repr = convert_float_to_quantized_int_repr(float(self.data_features[i][j]), eqn_simulation_dict['fractional_bits'], eqn_simulation_dict['total_bits'])
                            bin_str = convert_quantized_int_repr_to_binary_string(int_repr, eqn_simulation_dict['total_bits'])
                            s += 'Debugging of Run {} - Input Feature {}: {} | {} | {}\n'.format(i, j, float(self.data_features[i][j]), int_repr, bin_str)
                    
                    final_float_predict = convert_quantized_int_rep_to_float(final_class_predict, eqn_simulation_dict['total_bits'], eqn_simulation_dict['fractional_bits'], return_val='range')
                    s += 'Debugging of Run {} - Result as Unsigned Integer: {}\n'.format(i, final_class_predict)
                    s += 'Debugging of Run {} - Result as Binary String: {}\n'.format(i, final_binary_result_string)
                    s += 'Debugging of Run {} - Result as Float Range: {}\n\n'.format(i, final_float_predict)

                f.write(s)
                self.handler.log('debug', s[:-1]) # -1 to not print the last \n

        if constant_output_flag: 
            self.handler.log('warning', 'LogicSimulator - Simulation of {}: Logic only delivers constant output!'.format(file_name))
        return file_path

    
    def _evaluate_eqn_line(self, line): 
        '''
            This method evaluates a single equation line (string) from the eqn-file. 
            Args: 
                line: the line form the equation file (string)
            Returns: 
                Boolean value that states if the expression is True or False
        '''

        # handling AND-gates in the following
        if '*' in line: 
            operand1 = line.split(' = ')[-1].split(' *')[0]
            operand2 = line.split(' * ')[-1].split(';')[0]
            # now do the actual operation in four cases
            # case 1: operand1 is an inverter and operand2 is not 
            if ('!' in operand1) and not ('!' in operand2): 
                result = (not self.intermediate_results_dict[operand1[1:]]) and (self.intermediate_results_dict[operand2])
            # case 2: operand2 is an inverter and operand1 is not 
            elif not ('!' in operand1) and ('!' in operand2): 
                result = (self.intermediate_results_dict[operand1]) and (not self.intermediate_results_dict[operand2[1:]])
            # case 3: both operands are no inverters 
            elif not ('!' in operand1) and not ('!' in operand2): 
                result = (self.intermediate_results_dict[operand1]) and (self.intermediate_results_dict[operand2])
            # case 4: both operands are inverters 
            else: 
                result = (not self.intermediate_results_dict[operand1[1:]]) and (not self.intermediate_results_dict[operand2[1:]])
        
        # handling OR-gates in the following
        else: 
            operand1 = line.split(' = ')[-1].split(' +')[0]
            operand2 = line.split(' + ')[-1].split(';')[0]
            # now do the actual operation in four cases
            # case 1: operand1 is an inverter and operand2 is not 
            if ('!' in operand1) and not ('!' in operand2): 
                result = (not self.intermediate_results_dict[operand1[1:]]) or (self.intermediate_results_dict[operand2])
            # case 2: operand2 is an inverter and operand1 is not 
            elif not ('!' in operand1) and ('!' in operand2): 
                result = (self.intermediate_results_dict[operand1]) or (not self.intermediate_results_dict[operand2[1:]])
            # case 3: both operands are no inverters 
            elif not ('!' in operand1) and not ('!' in operand2): 
                result = (self.intermediate_results_dict[operand1]) or (self.intermediate_results_dict[operand2])
            # case 4: both operands are inverters 
            else: 
                result = (not self.intermediate_results_dict[operand1[1:]]) or (not self.intermediate_results_dict[operand2[1:]])

        return result