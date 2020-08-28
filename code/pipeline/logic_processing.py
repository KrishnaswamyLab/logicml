import numpy as np
import itertools
from graphviz import Source
import pickle
import argparse
from logicml.code.pipeline.utils import *
from logicml.code.pipeline.nn import *
from logicml.code.pipeline.handler import *
import math
import random

class LogicProcessing: 
    def __init__(self, handler, args=None): 
        '''
            The class defines all routines of logic within NNs.
            Args: 
                handler: handler of type Handler() for logging
                args: argparser arguments
        '''
        self.args = args

        if handler is None: 
            self.handler = Handler()
        else: 
            self.handler = handler
        
    def get_abc_execution_path(self): 
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

    def get_lgn_execution_path(self): 
        '''
            This method is a helper method to get the path to the LogicNet executable.
            NOTE: It assumes the Git-Structure that the executable is located one folder above this script and then in a subfolder called logicnet.
            NOTE: It also assumes that there are two executables: lgn for verbosity = info and lgn_verbose for verbosity = debug (compiled under different flags)
            Returns: 
                path to the LogicNet executable
        '''
        # get absolute path to the LogicNet executable
        # first get the directory of this python file
        lgn_exec_path = os.path.dirname(os.path.abspath(__file__))
        # now, go up one folder in this git and go to logicnet folder
        if str.lower(self.args.verbosity) == 'info': 
            lgn_exec_path = os.path.dirname(lgn_exec_path) + '/logicnet/lgn'
        else: # using the executable with print-outs
            lgn_exec_path = os.path.dirname(lgn_exec_path) + '/logicnet/lgn_verbose'
        return lgn_exec_path
    
    def write_logicnet_executing_shell_script(self, flist_path, folder_path, file_name, lutsize=None): 
        '''
            Writes the shell script for executing LogicNet on previously created .data files and flist-file. 
            Args: 
                flist_path: full path to where the flist file is located 
                folder_path: the path to the folder in which to store the shell script file (without last slash)
                file_name: the name of the file to store the data in (without ending)
                lutsize: LogicNet LUT-size, optional to do that from the outside (e.g. depending on module inputs) - if None args.lgn_lutsize parameter will be used
            Returns: 
                full path to the created shell script
        '''

        lgn_exec_path = self.get_lgn_execution_path()

        # creating the shell script file
        script_filepath = self.handler.path(folder_path) + '{}.sh'.format(file_name)

        with open(script_filepath, "w") as f:
            # NOTE: number of outputs is always one, due to training one LogicNet per bit - TODO: adapt, once we don't have bit by bit training anymore
            if lutsize is None: 
                lutsize = self.args.lgn_lutsize
            else: 
                assert isinstance(lutsize, int)
            f.write('{} {} {} {} {} {}'.format(lgn_exec_path, self.args.lgn_depth, self.args.lgn_width, lutsize, 1, flist_path))
        
        return script_filepath

    def write_verilog_to_aig_shell_script(self, verilog_file_path, folder_path, file_name, export_aig=True): 
        '''
            Writes the shell script for handing over a Verilog file to ABC, dumping it into AIG and processing it. 
            Args: 
                verilog_file_path: full path to where the final Verilog file is located 
                folder_path: the path to the folder in which to store the shell script file (including last slash)
                file_name: the name of the file to store the shell script in (without ending)
                export_aig: Boolean to set to True if the AIG should be saved in an .aig file
            Returns: 
                sequence of: the full path to the created shell script, the full path to the statistics file, the path to the eqn-file
        '''

        abc_exec_path = self.get_abc_execution_path()

        # creating the shell script file
        script_filepath = folder_path + '{}.sh'.format(file_name)

        with open(script_filepath, "w") as f:
            # opening up the interactive abc command line tool
            f.write('{} << COMMANDS\n'.format(abc_exec_path))
            # adding the commands that should be executed in there
            f.write('%read {}\n'.format(verilog_file_path))
            f.write('%blast\n')
            if self.args.aig_optimization: 
                if str.lower(self.args.aig_optimization_command) == 'dc2':
                    # less scalable but is guaranteed to never increase the size
                    if str.lower(self.args.verbosity) == 'debug':
                        optim_command = '&dc2 -v'
                    else: 
                        optim_command = '&dc2'
                elif str.lower(self.args.aig_optimization_command) == 'mfs':
                    # don't-care-based optimization on mapped AIGs
                    # NOTE: may work up to 1M nodes, if the logic is not too deep (under 100 AIG levels)
                    # if too deep, use command-line switches to restrict the depth of nodes considered --> use mfs_advanced
                    if str.lower(self.args.verbosity) == 'debug':
                        optim_command = '&mfs -w'
                    else: 
                        optim_command = '&mfs'
                elif str.lower(self.args.aig_optimization_command) == 'mfs_advanced':
                    # switches to restrict the depth of nodes before performing a don't care minimization 
                    if str.lower(self.args.verbosity) == 'debug':
                        optim_command = '&syn2 -w\n&if -K 6\n&mfs -w\n&st'
                    else: 
                        optim_command = '&syn2\n&if -K 6\n&mfs\n&st'
                else: # meaning if str.lower(self.args.aig_optimization_command) == 'syn2'
                    # &syn2 = most powerful exact synthesis / optimization (performs unmapping and mapping) 
                    # finds shared nodes, etc. but keeps original logic function - works well up to 10 mio. nodes, but may increase size
                    if str.lower(self.args.verbosity) == 'debug':
                        optim_command = '&syn2 -w'
                    else: 
                        optim_command = '&syn2'
                
                f.write('{}\n'.format(optim_command))
            
            # print statistics of AIG
            # if this comes after syn2, then this means it is the statistics after compressing the AIG
            statistics_pth = folder_path + file_name + '_aig_statistics.json'
            f.write('&ps -D {}\n'.format(statistics_pth))
            f.write('&put\n')

            # exporting the processed Verilog file:
            if self.args.abc_verilog_export:
                verilog_dump_pth = folder_path + '{}_abc.v'.format(file_name)
                f.write('write {}\n'.format(verilog_dump_pth))

            # exporting the eqn file for simulation: 
            eqn_path = folder_path + '{}_abc_equations.eqn'.format(file_name)
            f.write('write_eqn {}\n'.format(eqn_path))
            
            if export_aig: 
                aig_dump_pth = folder_path + '{}.aig'.format(file_name)
                f.write('write {}\n'.format(aig_dump_pth))
            # exit the interactive shell
            f.write('quit\n')
            f.write('COMMANDS\n')

        return script_filepath, statistics_pth, eqn_path

    def write_module_declaration_as_verilog(self, module_name, num_data_inputs, argmax_width, f): 
        '''
            A helper method that writes the preliminary part of the module in Verilog for the module declaration. 
            Args: 
                module_name: name that should be given to the Verilog module
                num_data_inputs: the number of data inputs (number of features from the data reader)
                argmax_width: bit width of the final argmax output after the last layer for classification (as many bits as needed for representation of number of classes)
                f: the python file object to which to write to
        '''

        f.write('// Quantization Scheme - Total Bits: {} - Fractional Bits: {}\n'.format(self.args.total_bits, self.args.fractional_bits))
        f.write('module {}(\n'.format(module_name))
            
        for i in range(num_data_inputs):
            f.write('    input wire signed [{}:0] final_in{},\n'.format(self.args.total_bits-1, i))       
        
        # NOTE: wire does not have to be signed, due to being the argmax final output wire
        f.write('    output wire [{}:0] final_out\n'.format(argmax_width-1))       
        f.write('    );\n\n')

    
    def concatenate_random_forest_bitwise_verilog_modules(self, num_data_inputs, folder_path, module_name): 
        '''
            Concatenates the different verilog modules from different Random Forest Verilog files and forms one flattened Verilog file and module.
            Hereby, it is meant that the modules that were trained for one bit each of the same outputting node are concatenated.
            Thus, this serves as a preparation step for the actual concatenation of modules of outputting nodes.
            Args: 
                num_data_inputs: the number of data inputs (number of features that the outputting node receives)
                folder_path: path to the folder to store the verilog file in (including last slash)
                module_name: name that should be given to the file and Verilog module
            Returns: 
                path to the final verilog file
        ''' 

        # the filepath for creating the final verilog file with the final module of the concatenated bit-wise modules 
        filepath = folder_path + module_name +'.v'
        
        # the folder in which the Verilog files for LogicNet are located
        rf_modules_path = self.handler.path('random_forest_files')

        with open(filepath, 'w') as f:

            # write preliminary part, i.e. module declaration: 
            f.write('// Quantization Scheme - Total Bits: {} - Fractional Bits: {}\n'.format(self.args.total_bits, self.args.fractional_bits))
            f.write('module {}(\n'.format(module_name))
            for i in range(num_data_inputs):
                f.write('    input signed [{}:0] {}_x{},\n'.format(self.args.total_bits-1, module_name, i))     
            f.write('    output signed [{}:0] {}_out\n'.format(self.args.total_bits-1, module_name))       
            f.write('    );\n\n')

            for i in range(self.args.total_bits): 
                identifier = module_name + '_bit{}'.format(i)
                current_module_path = rf_modules_path + '{}.v'.format(identifier)
                
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    // -------- starting bit {} of logic block {} from Random Forest --------\n'.format(i, module_name))   
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line) or (('input' in line) and not ('//' in line)) or (');' in line):
                            pass
                        elif (('output' in line) and not ('//' in line)): 
                            s = line.replace('output ', '')[:-1]+';\n'
                            f.write(s)
                        elif (('{}_bit{}_x_orig'.format(module_name, i) in line) and not ('input' in line)): 
                            orig_input_number = line.split('{}_bit{}_x_orig'.format(module_name, i))[-1].split(',')[0] 
                            name_string = module_name+'_bit{}_x{}'.format(i, orig_input_number)
                            s_prefix = '    wire [0:0] {}_{};\n'.format(name_string, i)
                            replace_string = module_name+'_x{}[{}]'.format(orig_input_number, i)
                            s_prefix += '    assign {}_{} = {};\n'.format(name_string, i, replace_string)
                            name_string = name_string+'_{}'.format(i)
                            s = line.replace('{}_bit{}_x_orig{}'.format(module_name, i, orig_input_number), name_string)
                            f.write(s_prefix+s)
                        else: 
                            f.write(line)

            # doing the concatenation of the outputs of each bit's LogicNet
            f.write('    // -------- final concatenated result of logic block {} from Random Forest --------\n'.format(module_name)) 
            s = '    assign {}_out = {} '.format(module_name, '{')
            for i in reversed(range(self.args.total_bits)): 
                s += '{}_bit{}_out{}'.format(module_name, i, ', ' if i > 0 else '')
            s += ' {};\n'.format('}')
            f.write(s)
            # final line 
            f.write('endmodule\n')

        return filepath
    
    
    def concatenate_random_forest_verilog_modules(self, num_data_inputs, first_layer_num_nodes, accumulated_truthtables_dict, folder_path, module_name): 
        '''
            Concatenates the different verilog modules from different Random Forest Verilog files and forms one flattened Verilog file and module.
            Args: 
                num_data_inputs: the number of data inputs (number of features from the data reader)
                first_layer_num_nodes: the number of nodes on the first layer
                accumulated_truthtables_dict: the dictionary that stored the activations from which the logic blocks have been created
                folder_path: path to the folder to store the verilog file in (including last slash)
                module_name: name that should be given to the file and Verilog module
            Returns: 
                path to the final verilog file
        ''' 

        # get the node-IDs of the last layer nodes 
        sorted_out_nodes_list = []
        for idx, k in enumerate(reversed(sorted(accumulated_truthtables_dict.keys()))): 
            if idx == 0: 
                last_layer = k.split('l')[-1].split('n')[0]
                sorted_out_nodes_list.append(k+'_rf') # NOTE: last layer nodes are come from Random Forest, therefore ending _rf
            else: 
                layer = k.split('l')[-1].split('n')[0]
                if layer == last_layer: 
                    sorted_out_nodes_list.append(k+'_rf') # NOTE: last layer nodes are come from Random Forest, therefore ending _rf
                else: 
                    break
        sorted_out_nodes_list = sorted(sorted_out_nodes_list)
        argmax_width = math.ceil(math.log2(len(sorted_out_nodes_list) + len(sorted_out_nodes_list)%2))
        
        filepath = folder_path + module_name +'.v'
        
        # the folder in which the Verilog files for the Random Forest are located
        rf_modules_path = self.handler.path('random_forest_files')

        # the folder in which the Verilog files for the Neural Network (first layer) are located
        nn_modules_path = self.handler.path('nn_logic_files')

        with open(filepath, 'w') as f:

            # write the preliminary part (the module declaration) to ignore all the following module declarations
            # NOTE: here we need the output line to be of the width of the argmax
            argmax_width = math.ceil(math.log2(len(sorted_out_nodes_list) + len(sorted_out_nodes_list)%2))
            self.write_module_declaration_as_verilog(module_name, num_data_inputs, argmax_width, f)
            
            # start concatenating the information from the first layer NN Verilog modules
            for i in range(first_layer_num_nodes): 
                current_module_path = nn_modules_path + 'l0n{}_nn.v'.format(i)
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    // -------- starting logic block from NN: l0n{}_nn --------\n'.format(i))   
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line) or (('input' in line) and not ('//' in line)) or (');' in line):
                            pass 

                        elif ('output' in line): 
                            s = line.replace('output ', '')[:-1] + ';\n'
                            f.write(s)
                        else: 
                            f.write(line)

            # now write all the other modules into the text file
            for k in sorted(accumulated_truthtables_dict.keys()):
                names, ins, outs = accumulated_truthtables_dict[k]
                current_module_path = rf_modules_path + k + '_rf.v'
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    //  -------- starting logic block from Random Forest: {} --------\n'.format(k))    
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line):
                            pass 

                        elif (');' in line): 

                            # the original module declaration is finished, meaning also the input and out wires of the modules are declared at this point  
                            # thus we can actually connect the input wire with the previous module output wires according to the connections in the network
                            for idx, n in enumerate(sorted(names)): 
                                s = '    assign {}_rf_x{} = '.format(k, idx)
                                if 'l0' in n: # first layer comes from neural network
                                    s += '{}_nn_out;\n'.format(n)
                                else: # next layers come from Random Forest
                                    s += '{}_rf_out;\n'.format(n)
                                f.write(s)

                        else: 
                            s = line
                            if (',' in line) and not ('=' in line): 
                                s = s.replace(',', ';')
                            if ('input' in line) and not ('//' in line): 
                                s = s.replace('input', 'wire')
                            if ('output' in line) and not ('//' in line): 
                                s = s.replace('output', 'wire')[:-1] + ';\n'
                            f.write(s)

            # final argmax and assignment of final result
            # NOTE: argmax only on last layer nodes and will be with bit width of as many bits as needed to represent the number of classes
            self.write_verilog_last_layer_argmax(sorted_out_nodes_list, f)
            # final line 
            f.write('endmodule\n')

        return filepath


    def concatenate_logicnet_bitwise_verilog_modules(self, num_data_inputs, folder_path, module_name): 
        '''
            Concatenates the different verilog modules from different LogicNet Verilog files and forms one flattened Verilog file and module.
            Hereby, it is meant that the modules that were trained for one bit each of the same outputting node are concatenated.
            Thus, this serves as a preparation step for the actual concatenation of modules of outputting nodes.
            Args: 
                num_data_inputs: the number of data inputs (number of features that the outputting node receives)
                folder_path: path to the folder to store the verilog file in (including last slash)
                module_name: name that should be given to the file and Verilog module
            Returns: 
                path to the final verilog file
        ''' 

        # the filepath for creating the final verilog file with the final module of the concatenated bit-wise modules 
        filepath = folder_path + module_name +'.v'
        
        # the folder in which the Verilog files for LogicNet are located
        lgn_modules_path = self.handler.path('logicnet_files')

        with open(filepath, 'w') as f:

            # write preliminary part, i.e. module declaration: 
            f.write('// Quantization Scheme - Total Bits: {} - Fractional Bits: {}\n'.format(self.args.total_bits, self.args.fractional_bits))
            f.write('module {}(\n'.format(module_name))
            for i in range(num_data_inputs):
                f.write('    input signed [{}:0] {}_x{},\n'.format(self.args.total_bits-1, module_name, i))
            f.write('    output signed [{}:0] {}_out\n'.format(self.args.total_bits-1, module_name))
            f.write('    );\n\n')

            for i in range(self.args.total_bits): 
                identifier = module_name + '_bit{}'.format(i)
                current_module_path = lgn_modules_path + '{}.v'.format(identifier)
                
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    // -------- starting bit {} of logic block {} from LogicNet --------\n'.format(i, module_name))   
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line) or (('input' in line) and not ('//' in line)) or (');' in line) or ('ABC' in line):
                            pass
                        elif (('output' in line) and not ('//' in line)): 
                            s = line.replace('output ', 'wire [0:0] ')[:-1]+';\n'
                            f.write(s)
                        elif (('{}_bit{}_x'.format(module_name, i) in line) and not ('input' in line)): 
                            s = line.replace('{}_bit{}_x['.format(module_name, i), module_name+'_x')
                            s = s.replace(']', '[{}]'.format(i))
                            f.write(s)
                        else: 
                            f.write(line)

            # doing the concatenation of the outputs of each bit's LogicNet
            f.write('    // -------- final concatenated result of logic block {} from LogicNet --------\n'.format(module_name)) 
            s = '    assign {}_out = {} '.format(module_name, '{')
            for i in reversed(range(self.args.total_bits)): 
                s += '{}_bit{}_y{}'.format(module_name, i, ', ' if i > 0 else '')
            s += ' {};\n'.format('}')
            f.write(s)
            # final line 
            f.write('endmodule\n')

        return filepath

    def write_verilog_last_layer_argmax(self, sorted_input_wires_list, f):
        '''
            Method to write an Verilog-argmax after an identity mapping to give the final class prediction.
            Args: 
                sorted_input_wires_list: the output-wires of the node-IDs in a sorted list that form the input wires to the argmax
                f: the opened Verilog-file to write to
        '''
        # NOTE: a few important points in the following
        # this method assumes the ouput wires to already be established from previous logic blocks, which should serve as input_wires to this block
        # those input wires always have the name <node_name>_out and always have the bit-width of self.num_total_bits
        # also do not create the output wire as this is also always created in the top module
        # when using the final output wire, use the bit-width that is needed to represent the number of classes (i.e. the number of inputs to the argmax)
        # the output is always and unsigned binary because we don't have negative classes 

        argmax_width = math.ceil(math.log2(len(sorted_input_wires_list) + len(sorted_input_wires_list)%2))

        # format(x, 'b').zfill(n) 

        # writing the intialization
        f.write('\n    //  -------- starting final argmax --------\n') 
        f.write('    // initialization with max = {}_out\n'.format(sorted_input_wires_list[0]))
        f.write('    wire [{}:0] final_argmax0;\n'.format(argmax_width-1))
        f.write('    assign final_argmax0 = {}\'b{};\n'.format(argmax_width, format(0, 'b').zfill(argmax_width)))
        f.write('    wire signed [{}:0] final_max0;\n'.format(self.args.total_bits-1))
        f.write('    assign final_max0 = {}_out;\n'.format(sorted_input_wires_list[0]))

        # writing the blocks of comparisons
        for i in range(1, len(sorted_input_wires_list)): 
            out_node_name = sorted_input_wires_list[i] + '_out'
            f.write('    // comparing max against {}\n'.format(out_node_name))
            f.write('    wire final_c{};\n'.format(i))
            # NOTE: in the following it is a question of the argmax implementation, you can either take > or >=, which takes the smaller or bigger index when two maximum values are the same
            # this is a problem especially when two points for the final prediction are close to the decision boundary (e.g. 0.52 and 0.48)
            # because in this case they often map onto the same binary value which means a clear separation of the classes is not possible anymore
            # probably take > as convention for the argmax because this is numpy convention
            if self.args.argmax_greater_equal: 
                f.write('    assign final_c{} = {} >= final_max{};\n'.format(i, out_node_name, i-1)) 
            else: 
                f.write('    assign final_c{} = {} > final_max{};\n'.format(i, out_node_name, i-1)) 
            f.write('    wire signed [{}:0] final_max{};\n'.format(self.args.total_bits-1, i))
            f.write('    assign final_max{} = final_c{} ? {} : final_max{};\n'.format(i, i, out_node_name, i-1))
            f.write('    wire [{}:0] index{};\n'.format(argmax_width-1, i))
            f.write('    assign index{} = {}\'b{};\n'.format(i, argmax_width, format(i, 'b').zfill(argmax_width)))
            f.write('    wire [{}:0] final_argmax{};\n'.format(argmax_width-1, i))
            f.write('    assign final_argmax{} = final_c{} ? index{} : final_argmax{};\n'.format(i, i, i, i-1))
        
        # assigning the final result to the output variable
        f.write('\n    //  -------- final assignment --------\n') 
        f.write('    assign final_out = final_argmax{};\n'.format(len(sorted_input_wires_list)-1))



    def concatenate_logicnet_verilog_modules(self, num_data_inputs, first_layer_num_nodes, accumulated_truthtables_dict, folder_path, module_name): 
        '''
            Concatenates the different verilog modules from different LogicNet Verilog files and forms one flattened Verilog file and module.
            Args: 
                num_data_inputs: the number of data inputs (number of features from the data reader)
                first_layer_num_nodes: the number of nodes on the first layer
                accumulated_truthtables_dict: the dictionary that stored the activations from which the logic blocks have been created
                folder_path: path to the folder to store the verilog file in (including last slash)
                module_name: name that should be given to the file and Verilog module
            Returns: 
                path to the final verilog file
        ''' 
        
        # get the node-IDs of the last layer nodes 
        sorted_out_nodes_list = []
        for idx, k in enumerate(reversed(sorted(accumulated_truthtables_dict.keys()))): 
            if idx == 0: 
                last_layer = k.split('l')[-1].split('n')[0]
                sorted_out_nodes_list.append(k+'_lgn') # NOTE: last layer nodes are come from LogicNet, therefore ending _lgn
            else: 
                layer = k.split('l')[-1].split('n')[0]
                if layer == last_layer: 
                    sorted_out_nodes_list.append(k+'_lgn') # NOTE: last layer nodes are come from LogicNet, therefore ending _lgn
                else: 
                    break
        sorted_out_nodes_list = sorted(sorted_out_nodes_list)


        filepath = folder_path + module_name +'.v'
        
        # the folder in which the Verilog files for LogicNet are located
        lgn_modules_path = self.handler.path('logicnet_files')

        # the folder in which the Verilog files for the Neural Network (first layer) are located
        nn_modules_path = self.handler.path('nn_logic_files')

        with open(filepath, 'w') as f:

            # write the preliminary part (the module declaration) to ignore all the following module declarations
            # NOTE: here we need the output line to be of the width of the argmax
            argmax_width = math.ceil(math.log2(len(sorted_out_nodes_list) + len(sorted_out_nodes_list)%2))
            self.write_module_declaration_as_verilog(module_name, num_data_inputs, argmax_width, f)
            
            # start concatenating the information from the first layer NN Verilog modules
            for i in range(first_layer_num_nodes): 
                current_module_path = nn_modules_path + 'l0n{}_nn.v'.format(i)
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    // -------- starting logic block from NN: l0n{}_nn --------\n'.format(i))   
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line) or (('input' in line) and not ('//' in line)) or (');' in line):
                            pass 

                        elif ('output' in line): 
                            s = line.replace('output ', '')[:-1] + ';\n'
                            f.write(s)
                        else: 
                            f.write(line)

            # now write all the other modules into the text file
            for k in sorted(accumulated_truthtables_dict.keys()):
                names, ins, outs = accumulated_truthtables_dict[k]
                current_module_path = lgn_modules_path + k + '_lgn.v'
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    //  -------- starting logic block from LogicNet: {} --------\n'.format(k))    
                    for line in lines: 
                        if ('module' in line) or ('ABC' in line) or ('endmodule' in line):
                            pass 

                        elif (');' in line): 

                            # the original module declaration is finished, meaning also the input and out wires of the modules are declared at this point  
                            # thus we can actually connect the input wire with the previous module output wires according to the connections in the network
                            for idx, n in enumerate(sorted(names)): 
                                s = '    assign {}_lgn_x{} = '.format(k, idx)
                                if 'l0' in n: # first layer comes from neural network
                                    s += '{}_nn_out;\n'.format(n)
                                else: # next layers come from LogicNet
                                    s += '{}_lgn_out;\n'.format(n)
                                f.write(s)

                        else: 
                            s = line
                            if (',' in line) and not ('=' in line): 
                                s = s.replace(',', ';')
                            if 'input' in line: 
                                s = s.replace('input', 'wire')
                            if 'output' in line: 
                                s = s.replace('output', 'wire')[:-1] + ';\n'
                            f.write(s)

            # final argmax and assignment of final result
            # NOTE: argmax only on last layer nodes and will be with bit width of as many bits as needed to represent the number of classes
            self.write_verilog_last_layer_argmax(sorted_out_nodes_list, f)
            # final line 
            f.write('endmodule\n')

        return filepath

    def concatenate_nn_verilog_modules(self, num_data_inputs, first_layer_num_nodes, accumulated_truthtables_dict, folder_path, module_name): 
        '''
            Concatenates the different verilog modules from different Neural Network Verilog files and forms one flattened Verilog file and module.
            Args: 
                num_data_inputs: the number of data inputs (number of features from the data reader)
                first_layer_num_nodes: the number of nodes on the first layer
                accumulated_truthtables_dict: the dictionary that stored the activations from which the logic blocks have been created
                folder_path: path to the folder to store the verilog file in (including last slash)
                module_name: name that should be given to the file and Verilog module
            Returns: 
                path to the final verilog file
        ''' 

        # get the node-IDs of the last layer nodes 
        sorted_out_nodes_list = []
        for idx, k in enumerate(reversed(sorted(accumulated_truthtables_dict.keys()))): 
            if idx == 0: 
                last_layer = k.split('l')[-1].split('n')[0]
                sorted_out_nodes_list.append(k+'_nn') # NOTE: last layer nodes are come from Neural Network, therefore ending _nn
            else: 
                layer = k.split('l')[-1].split('n')[0]
                if layer == last_layer: 
                    sorted_out_nodes_list.append(k+'_nn') # NOTE: last layer nodes are come from Neural Network, therefore ending _nn
                else: 
                    break
        sorted_out_nodes_list = sorted(sorted_out_nodes_list)
        argmax_width = math.ceil(math.log2(len(sorted_out_nodes_list) + len(sorted_out_nodes_list)%2))
        
        filepath = folder_path + module_name +'.v'
        
        # the folder in which the Verilog files for the Neural Network are located
        nn_modules_path = self.handler.path('nn_logic_files')

        with open(filepath, 'w') as f:

            # write the preliminary part (the module declaration) to ignore all the following module declarations
            self.write_module_declaration_as_verilog(module_name, num_data_inputs, argmax_width, f)
            
            # start concatenating the information from the first layer NN Verilog modules - doing this separately
            for i in range(first_layer_num_nodes): 
                current_module_path = nn_modules_path + 'l0n{}_nn.v'.format(i)
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('\n    // -------- starting logic block from NN: l0n{}_nn --------\n'.format(i))   
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line) or (('input' in line) and not ('//' in line)) or (');' in line):
                            pass 

                        elif ('output' in line): 
                            s = line.replace('output ', '')[:-1] + ';\n'
                            f.write(s)
                        else: 
                            f.write(line)

            # now write all the other modules into the text file
            for k in sorted(accumulated_truthtables_dict.keys()):
                names, ins, outs = accumulated_truthtables_dict[k]
                names = sorted(names)
                current_module_path = nn_modules_path + k + '_nn.v'
                with open(current_module_path, 'r') as current_module_f: 
                    lines = current_module_f.readlines()
                    f.write('    //  -------- starting logic block from NN: {} --------\n'.format(k))    
                    for line in lines: 
                        if ('module' in line) or ('endmodule' in line) or (('input' in line) and not ('//' in line)) or (');' in line):
                            pass 

                        elif ('output' in line) and not ('//' in line): 
                            s = line.replace('output ', '')[:-1] + ';\n'
                            f.write(s)
                        
                        elif ('assign {}_nn_in'.format(k) in line): # doing the actual connection to previous modules by replacing the input_names
                            input_number = int(line.split('{}_nn_x'.format(k))[-1].split(';')[0])
                            # NOTE: here no support of skip-connections yet - do it when needed
                            s = line.replace('{}_nn_x{}'.format(k, input_number), '{}_nn_out'.format(names[input_number]))
                            f.write(s)
                        
                        else: 
                            f.write(line)

            # final argmax and assignment of final result
            # NOTE: argmax only on last layer nodes and will be with bit width of as many bits as needed to represent the number of classes
            self.write_verilog_last_layer_argmax(sorted_out_nodes_list, f)

            # final line 
            f.write('endmodule\n')

        return filepath


    def neural_net_dump_verilog(self, net, folder_path, suffix='nn', only_first_layer=False):
        '''
            Method to dump the a neural network structure into Verilog modules.
            One module per node in the network. 
            self.Args: 
                net: the neural network pytorch object
                folder_path: path to the folder to store the verilog file in (without last slash)
                suffix: name that should be appended as suffix to the modules (without any ending)
                only_first_layer: Boolean to set to True if only the first layer of the NN should be turned into Verilog modules
            Returns: 
                dict of dumped Verilog files of created modules (keys = nodeIDs, values = filepaths)
        '''
        return_files_dict = {}

        if only_first_layer: 
            max_number = 1
        else: 
            max_number = net.hidden_num_layers+1

        for layer_number in range(max_number): 
            if layer_number == net.hidden_num_layers: # meaning the final layer
                weight_key = 'layers.Final_Stage{}.fc_final.weight'.format(layer_number)
                if self.args.bias: 
                    bias_key = 'layers.Final_Stage{}.fc_final.bias'.format(layer_number)
            else: 
                weight_key = 'layers.FC_Stage{}.fc{}.weight'.format(layer_number, layer_number)
                if self.args.bias: 
                    bias_key = 'layers.FC_Stage{}.fc{}.bias'.format(layer_number, layer_number)

            if self.args.cuda: 
                full_weights = net.state_dict()[weight_key].clone().cpu().detach().numpy()
                if self.args.bias: 
                    full_biases = net.state_dict()[bias_key].clone().cpu().detach().numpy()
            else: 
                full_weights = net.state_dict()[weight_key].clone().detach().numpy()
                if self.args.bias: 
                    full_biases = net.state_dict()[bias_key].clone().detach().numpy()

            # NOTE: modeling each node of the layer separately as one Verilog module 
            # this is done due to giving the highest flexibility layer on in the concatenation of the modules
            # this sometimes also means that a conversion of unsigned and signed outputs is done within the modules...
            # ...that might not be necessary when only stacking NN modules, but it is when stacking with modules from random forest or LogicNet
            for node_number in range(full_weights.shape[0]):
                self.handler.log('debug', 'NN - Processing Logic Block: l{}n{}'.format(layer_number, node_number))
                module_name = 'l{}n{}_{}'.format(layer_number, node_number, suffix)
                weights = full_weights[node_number].tolist()
                if self.args.bias: 
                    bias = full_biases.tolist()[node_number]
                else: 
                    bias = None # anyway not used

                # we need to treat nodes of the first layer differently than all other nodes
                if layer_number == 0: 
                    num_inputs = net.num_features
                    verilog_file = self.neural_net_dump_verilog_module(num_inputs, folder_path, module_name, weights, bias, activation_type='relu', first_layer=True) 
                # we need to treat nodes of the last layer differently than all other nodes
                elif layer_number == net.hidden_num_layers: 
                    # NOTE: here we are not able to model skip-connections yet
                    # TODO: change once we also want to model skip-connections
                    num_inputs = net.hidden_out_nodes[layer_number-1]
                    verilog_file = self.neural_net_dump_verilog_module(num_inputs, folder_path, module_name, weights, bias, activation_type='identity', first_layer=False)
                else: # intermediate nodes 
                    # NOTE: here we are not able to model skip-connections yet
                    # TODO: change once we also want to model skip-connections
                    num_inputs = net.hidden_out_nodes[layer_number-1]
                    verilog_file = self.neural_net_dump_verilog_module(num_inputs, folder_path, module_name, weights, bias, activation_type='relu', first_layer=False)
                
                return_files_dict['l{}n{}'.format(layer_number, node_number)] = verilog_file
        return return_files_dict



    def neural_net_dump_verilog_module(self, num_inputs, folder_path, module_name, weights, bias, activation_type='relu', first_layer=False):
        '''
            Helper method to dump the a neural network structure into Verilog modules.
            Outputs the logic for one specific node. 
            Args: 
                num_inputs: the number of inputs to the module
                folder_path: path to the folder to store the verilog file in (without last slash)
                module_name: name that should be given to the file, the variable prefixes and the verilog module (without any ending)
                weights: a list of floats that are the weights of this specific outputting nodes
                bias: a float number that is the bias of this specific outputting node 
                activation_type: identifier for which type of the activation function should be modeled: identity or ReLU
                first_layer: Boolean to set to True when the modeled node is part of the first layer
            Returns: 
                path to the dumped Verilog file
        '''

        # start writing the actual module declaration
        filepath = folder_path + '/' + module_name +'.v'
        with open(filepath, 'w') as f:
            f.write('// Quantization Scheme - Total Bits: {} - Fractional Bits: {}\n'.format(self.args.total_bits, self.args.fractional_bits))
            f.write('module {}(\n'.format(module_name))
            
            for i in range(num_inputs):
                if first_layer: 
                    f.write('    input wire signed [{}:0] final_in{}{}\n'.format(self.args.total_bits-1, i, ','))
                else: 
                    f.write('    input wire signed [{}:0] {}_x{}{}\n'.format(self.args.total_bits-1, module_name, i, ','))
            
            f.write('    output wire signed [{}:0] {}_out\n'.format(self.args.total_bits-1, module_name))       
            f.write('    );\n')

            # starting with the actual weight and input data multiplications and summing up
            for i in range(num_inputs):
                f.write('\n    // {} - handling input number: {}\n'.format(module_name, i))
                f.write('    wire signed [{}:0] {}_in{};\n'.format(self.args.total_bits-1, module_name, i))
                if first_layer:
                    f.write('    assign {}_in{} = final_in{};\n'.format(module_name, i, i))
                else:  
                    f.write('    assign {}_in{} = {}_x{};\n'.format(module_name, i, module_name, i))
                f.write('    wire signed [{}:0] {}_weight{};\n'.format(self.args.total_bits-1, module_name, i))
                binarized = convert_float_to_quantized_int_repr(weights[i], self.args.fractional_bits, self.args.total_bits)
                binarized = convert_quantized_int_repr_to_binary_string(binarized, self.args.total_bits)
                f.write('    assign {}_weight{} = {}\'b{}; // weight: {}\n'.format(module_name, i, self.args.total_bits, binarized, weights[i]))

                
                # starting with the actual multiplication
                # NOTE: we assume that the multiplicator logic has the doubled number of total quantization bits 
                f.write('    // the actual multiplication \n')
                f.write('    wire signed [{}:0] {}_mult{};\n'.format(2*self.args.total_bits-1, module_name, i))
                f.write('    assign {}_mult{} = {}_weight{} * {}_in{};\n'.format(module_name, i, module_name, i, module_name, i))
                
                # starting with the accumulation
                # NOTE: we assume that the multiplicator logic has the tripled number of total quantization bits 
                f.write('    // accumulation step \n')
                f.write('    wire signed [{}:0] {}_acc{};\n'.format(3*self.args.total_bits-1, module_name, i))
                # extend the multiplication result to have the same size of the accumulator 
                f.write('    wire signed [{}:0] {}_mult_extended{};\n'.format(3*self.args.total_bits-1, module_name, i))
                f.write('    assign {}_mult_extended{} = {} {}{}{}{}_mult{}[{}]{}{}, {}_mult{} {};\n'.format(module_name, i, '{', '{', self.args.total_bits, '{', module_name, i, 2*self.args.total_bits-1, '}', '}', module_name, i, '}'))
                if i == 0: # first step is different - we need to do the initalization
                    f.write('    assign {}_acc{} = {}_mult_extended{};\n'.format(module_name, i, module_name, i))
                else: 
                    f.write('    assign {}_acc{} = {}_acc{} + {}_mult_extended{};\n'.format(module_name, i, module_name, i-1, module_name, i))


            if self.args.bias: 
                # adding the bias - is quantized to num_total_bits, but needs to be brought to triple of this size, due to need of being of same size as accumulator
                f.write('\n    // adding the bias\n')
                f.write('    wire signed [{}:0] {}_bias;\n'.format(3*self.args.total_bits-1, module_name))
                binarized = convert_float_to_quantized_int_repr(bias, self.args.fractional_bits, self.args.total_bits)
                binarized = convert_quantized_int_repr_to_binary_string(binarized, self.args.total_bits)
                # NOTE: here we need to add a couple of zeros:  
                # - in front (for the additional int-bits introduced by the multiplicator and the accumulator)
                # - in the end (for the additional fractional bits introduced by the multiplicator)
                # NOTE: MSB filling here with MSB of bias binarization (i.e. arithmetic extension) 
                f.write('    assign {}_bias = {}\'b{}{}{}; // bias: {}\n'.format(module_name, 3*self.args.total_bits, (2*self.args.total_bits-self.args.fractional_bits)*binarized[0], binarized, self.args.fractional_bits*'0', bias))                
                f.write('    wire signed [{}:0] {}_acc{};\n'.format(3*self.args.total_bits-1, module_name, num_inputs))
                f.write('    assign {}_acc{} = {}_acc{} + {}_bias;\n'.format(module_name, num_inputs, module_name, num_inputs-1, module_name))
                
            else: # handling argparser option that no bias is used - Just do a renaming for the purpose of being consistent with the upcoming modules
                    f.write('\n    // leaving out the bias, but doing a renaming for consistency within the activation function\n')
                    f.write('    wire signed [{}:0] {}_acc{};\n'.format(3*self.args.total_bits-1, module_name, num_inputs))
                    f.write('    assign {}_acc{} = {}_acc{};\n'.format(module_name, num_inputs, module_name, num_inputs-1)) 

            # now we need to bring the accumulator result back to the original size of total_bits
            # doing this by shifting the additonal fractional bits to the right and then do a potential clipping 
            f.write('\n    // bringing accumulator result to correct size by shifting and clipping\n')
            f.write('    wire signed [{}:0] {}_final_acc_shifted;\n'.format(3*self.args.total_bits-1, module_name))
            f.write('    assign {}_final_acc_shifted = {}_acc{} >> {};\n'.format(module_name, module_name, num_inputs, self.args.fractional_bits)) 
            f.write('    wire signed [{}:0] {}_acc_clipped;\n'.format(self.args.total_bits-1, module_name))
            f.write('    assign {}_acc_clipped = {}_final_acc_shifted[{}:0];\n'.format(module_name, module_name, self.args.total_bits-1)) 

            # handling the RELU 
            if activation_type == 'relu': 
                f.write('\n    // ReLU activation function: f(x) = max(0, x)\n')
                f.write('    // doing the comparison against 0\n')
                f.write('    wire {}_relu_c;\n'.format(module_name))
                f.write('    wire signed [{}:0] {}_relu_comp;\n'.format(self.args.total_bits-1, module_name))
                f.write('    assign {}_relu_comp = {}\'b{};\n'.format(module_name, self.args.total_bits, self.args.total_bits*'0'))
                f.write('    assign {}_relu_c = {}_acc_clipped > {}_relu_comp;\n'.format(module_name, module_name, module_name)) # NOTE: >= or > --> definition of comparison can make a difference?
                f.write('\n    // final assignment of module {}\n'.format(module_name))
                f.write('    assign {}_out = {}_relu_c ? {}_acc_clipped : {}\'b{};\n'.format(module_name, module_name, module_name, self.args.total_bits, self.args.total_bits*'0'))

            else: # meaning identity mapping 
                f.write('\n    // final assignment of module {}\n'.format(module_name))
                f.write('    assign {}_out = {}_acc_clipped;\n'.format(module_name, module_name))

            f.write('endmodule')
        return filepath
        

    def logic_to_interpretable_equation_report(self, logic_file_path, input_names, output_name, folder_path, file_name, in_key='final_in', out_key='final_out'):
        '''
            This method takes a Verilog-file, blif-file or eqn-file that descibes the logic and derives a textfile report that is interpretable.
            This means that the equations and dependencies of the variables and AIG-nodes are reported. 
            The method can either be called on the full logic or on single logic modules. 
            Args: 
                logic_file_path: full path to either a Verilog file that encodes the logic or an already create eqn file
                input_names: list of input variables to the logic (can either be IDs from previous nodes or names of data features) - must be in right order
                output_name: name of the final output of the logic (like variable name)
                folder_path: folder in which the the created file should be stored (including last slash)
                file_name: name that should be given to the textfile (without ending)
                in_key: the output variable identifier in the Verilog module (e.g. final_in for full logic or first layer NN modules, <node>_nn_x for NN-translation; <node>_rf_x for Random Forest, <>_lgn_x for LogicNet)
                out_key: the output variable identifier in the Verilog module (e.g. final_out for full logic; <node>_rf_out for Random Forest, <>_lgn_out for LogicNet, <>_nn_out for NN-translation)
        ''' 
        assert ('.v' in logic_file_path) or ('.eqn' in logic_file_path) or ('.blif' in logic_file_path), 'Method can only handle Verilog-, blif- or eqn-files.'
        eqn_file_flag = '.eqn' in logic_file_path # means this flag is True when the handed over logic file is an eqn-file and needs no pre-processing
        final_report_path = folder_path + '{}_equation_report.txt'.format(file_name)

        if not eqn_file_flag: # i.e. Verilog-file we first have to create an eqn-file

            abc_exec_path = self.get_abc_execution_path()

            # creating the shell script file
            script_filepath = folder_path + '{}_report_eqn_creation.sh'.format(file_name)
            with open(script_filepath, "w") as f:
                # opening up the interactive abc command line tool
                f.write('{} << COMMANDS\n'.format(abc_exec_path))
                # adding the commands that should be executed in there
                if '.v' in logic_file_path: 
                    f.write('%read {}\n'.format(logic_file_path))
                    f.write('%blast\n')
                    f.write('&put\n')
                else: # meaning blif file
                    f.write('read {}\n'.format(logic_file_path))                     
                # exporting the eqn file and overwriting the logic_file_path
                logic_file_path = folder_path + '{}_eqn_report.eqn'.format(file_name)
                f.write('write_eqn {}\n'.format(logic_file_path))
                f.write('quit')
                f.write('COMMANDS\n')

            # executing the created shell script
            os.system('source {}'.format(script_filepath))
            # delete the shell script after is was executed
            os.system('rm {}'.format(script_filepath))

        # opening the file to which the report should be written 
        with open(final_report_path, 'w') as f: 
            # write declarations
            f.write('Logic Report: {}\n'.format(file_name))
            f.write('----------------------------------\n\n')
            
            f.write('Inputs:\n')
            f.write('----------------------------------\n\n')
            for idx, inp in enumerate(input_names): 
                f.write('Input {}:\t{}\n'.format(idx, inp))
            
            # f.write('----------------------------------\n')
            f.write('\nOutputs:\n')
            f.write('----------------------------------\n\n')
            f.write('Output:\t{}\n'.format(output_name))

            f.write('\nEquations:\n')
            f.write('----------------------------------\n\n')

            used_inputs_set = set()

            # reading the eqn file 
            with open(logic_file_path, 'r') as f_read: 
                lines = f_read.readlines()
                for line in lines: 
                    if '=' in line and not ('ORDER' in line): # means equation: 
                        line = line.replace('*', 'AND')
                        line = line.replace('+', 'OR')
                        line = line.replace('!', 'NOT ')
                        line = line.replace(out_key, output_name)

                        if in_key in line: 
                            # replacing in_key by the corresponding feature name from the input_names list
                            final_in_splits = line.split(in_key)
                            for i in range(1, len(final_in_splits)): 
                                if '[' in final_in_splits[i]: 
                                    feature_idx = int(final_in_splits[i].split('[')[0])
                                    used_inputs_set.add(feature_idx)
                                    line = line.replace('{}{}'.format(in_key, feature_idx), input_names[feature_idx])
                        f.write(line)

            if len(used_inputs_set) != len(input_names): # means not all inputs were used - write a warning
                f.write('\nWARNING: No Dependency on Inputs:\n')
                f.write('----------------------------------\n\n')
                for idx, inp in enumerate(input_names):
                    if idx not in used_inputs_set: 
                        f.write('Input {}:\t{}\n'.format(idx, inp))


        if not eqn_file_flag: # delete the created eqn-file since it is not needed anymore
            os.system('rm {}'.format(logic_file_path))

        # TODO: be able to read such a created report in another method and derive a graphviz visualization? only when not too big?