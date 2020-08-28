import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot, make_dot_from_trace
from graphviz import Source
import pydot
import numpy as np
import os
from array import *

def convert_float_to_quantized_int_repr(float_value, num_fract_bits, num_total_bits): 
    '''
        This method does the conversion of a float number into a representation that represents it in the binary way of the quantization scheme as int that still needs to be turned into a binary string. 
        Hereby, the sign of the float number is also evaluated, thus the binary string is a two-complement number. 
        Also, a saturation is applied, i.e.: 
            - if a value is smaller than the smallest possible value that can be represented by the binary string, the smallest possible value is returned
            - if a value is bigger than the biggest possible value that can be represented by the binary string, the biggest possible value is returned
        Args: 
            float_value: the float value to convert
            num_fact_bits: the number of fractional bits from the total bits (i.e. the number of bits that should be used to represent that part behind the comma)
            num_total_bits: the total number of bits wanted 
        Returns: 
            the quantized representation of the float value as two-complement with a length of num_total_bits as int value
            NOTE: this means that turning this number into a binary string is what gives us the final binary representation
    '''

    # NOTE: A N-bit two’s-complement system can represent every integer in the range -2^{N-1} to +2^{N-1}-1

    # TODO: more assertions ?
    assert num_fract_bits < num_total_bits, 'The number of fractional bits [{}] needs to be smaller than the number of total bits [{}] to do a conversion of a float number into a binary string.'.format(num_fract_bits, num_total_bits)
    num_int_bits = num_total_bits - num_fract_bits
    assert type(float_value) == float

    result = (1 << num_fract_bits) * float_value 
    result_int = int(result)
    
    result_int = min(largest_signed_int(num_total_bits), result_int)
    result_int = max(smallest_signed_int(num_total_bits), result_int)

    result_int = result_int & largest_unsigned_int(num_total_bits)
    assert result_int >= 0 # unsigned ints from here on (i.e. a positive int-number )
    return result_int

def convert_quantized_int_repr_to_binary_string(int_value, num_total_bits): 
    '''
        This method does the conversion of an int number into its binary string representation.
        Args: 
            int_value: the int value to convert
            num_total_bits: the total number of bits wanted 
        Returns: 
            the binary string representation of the int value
    '''
    assert type(int_value) == int
    return format(int_value, 'b').zfill(num_total_bits)


def largest_unsigned_int(num_total_bits): 
    '''
        Args: 
            num_total_bits: the number of bits wanted 
        Returns: 
            the largest unsigned integer number that can be represented with the given number of total bits
    '''
    return (1 << (num_total_bits))-1

def largest_signed_int(num_total_bits): 
    '''
        Args: 
            num_total_bits: the number of bits wanted 
        Returns: 
            the largest signed (2-complement) integer number that can be represented with the given number of total bits
    '''
    return (1 << (num_total_bits-1))-1

def smallest_signed_int(num_total_bits): 
    '''
        Args: 
            num_total_bits: the number of bits wanted 
        Returns: 
            the smallest signed (2-complement) integer number that can be represented with the given number of total bits
    '''
    return -(1 << (num_total_bits-1))

def convert_binary_string_to_quantized_int_repr(binary_string): 
    '''
        This method takes a binary string representation and turns it back into a quantized int representation.
        Args: 
            binary_string: the binary string value
        Returns: 
            integer value that is represented by the binary string
    '''
    return int(binary_string, 2)

def convert_quantized_int_rep_to_float(integer_rep, num_total_bits, num_fract_bits, return_val='range'): 
    '''
        This method takes an int representation of the quantization scheme and turns it back into an according float value.
        However, there is not just one float value to which it can be mapped to, but instead a range. 
        Thus, there is the option to gain the maximum, the minimum or the middle of the range or the full range itself.
        Args: 
            integer_rep value: the integer value according to quantization scheme 
            num_total_bits: number of total bits
            num_fract_bits: number of fractional bits: 
            return_val: decided if the maxmimum (max) / minimum (min) / middle (mid) / or full range (range) as sequence of the float-value range should be returned
        Returns: 
            float representation (either single value or sequence in case of full_range)

    '''
    assert return_val in ['max', 'mid', 'min', 'range'], 'Reverse binarization only accepts range, max, min or mid as parameters to return from the full value range.'
    negative_number = integer_rep > largest_signed_int(num_total_bits)
    if negative_number: 
        integer_rep = largest_unsigned_int(num_total_bits) - integer_rep + 1 
    result1 = float(integer_rep) / (1 << num_fract_bits)
    result2 = float(integer_rep) / ((1 << num_fract_bits)-1)
    if negative_number: 
        result1 = -1.0 * result1
        result2 = -1.0 * result2
    
    if negative_number: 
        if return_val == 'range':
            return (result2, result1)
        if return_val == 'min': 
            return result2
        elif return_val == 'max': 
            return result1
        else: 
            return ((result1 - result2) / 2.0) + result2
    else: 
        if return_val == 'range':
            return (result1, result2)
        elif return_val == 'min': 
            return result1
        elif return_val == 'max': 
            return result2
        else: 
            return ((result2 - result1) / 2.0) + result1


def process_torch_tensors_to_numpy_lists_bin_str_repr(tensors_list, cuda, num_fract_bits, num_total_bits, activations_output_mode=0): 
    '''
        Takes activations or outputs as torch tensors, quantizes them to binary strings and transforms them to numpy arrays.
        This method is therefore a helper for e.g. the bitwise LogicNet or Random Forest learning. 
        Args: 
            tensors_list: list of torch tensors (that form the activations) to be transformed
            cuda: True if torch tensors are cuda tensors
            num_fact_bits: the number of fractional bits from the total bits (i.e. the number of bits that should be used to represent that part behind the comma)
            num_total_bits: the total number of bits wanted 
            activations_output_mode: 0 if net activations are processed, 1 if net ouputs are processed
        Returns: 
            list of tensors transformed to numpy arrays for further processing (NOTE: the values in the arrays are strings )
    '''
    l = []
    for a in tensors_list: 
        if cuda: 
            a = a.clone().cpu().detach().numpy()
        else: 
            a = a.clone().detach().numpy()
        
        a = np.array(a, dtype=np.float)
        x = np.zeros(a.shape, dtype=np.str)
        x = np.array(x, dtype='object') # needed otherwise the binary strings will be cut shorter from numpy which is unwanted
        for idx in np.ndindex(*a.shape): # iterate over all possible indices and apply the quantization
            int_value = convert_float_to_quantized_int_repr(float(a[idx]), num_fract_bits, num_total_bits) # getting the int representation
            x[idx] = convert_quantized_int_repr_to_binary_string(int_value, num_total_bits) # getting the binary string of the int representation
        l.append(x)
    return l


def process_torch_tensors_to_numpy_lists_float_repr(tensors_list, cuda, num_fract_bits, num_total_bits, activations_output_mode=0): 
    '''
        Takes activations or outputs as torch tensors and transforms them to float numpy arrays.
        This method is therefore a helper for e.g. non-bitwise Random Forest learning. 
        Args: 
            tensors_list: list of torch tensors (that form the activations) to be transformed
            cuda: True if torch tensors are cuda tensors
            num_fact_bits: the number of fractional bits from the total bits (i.e. the number of bits that should be used to represent that part behind the comma)
            num_total_bits: the total number of bits wanted 
            activations_output_mode: 0 if net activations are processed, 1 if net ouputs are processed
        Returns: 
            list of tensors transformed to numpy arrays for further processing (NOTE: the values in the arrays are strings )
    '''
    l = []
    for a in tensors_list: 
        if cuda: 
            a = a.clone().cpu().detach().numpy()
        else: 
            a = a.clone().detach().numpy()
        
        a = np.array(a, dtype=np.float)
        l.append(a)
    return l


def process_torch_tensors_to_numpy_lists_bin_int_repr(tensors_list, cuda, num_fract_bits, num_total_bits, activations_output_mode=0): 
    '''
        Takes activations or outputs as torch tensors, quantizes them to an equivalent int representation and transforms them to numpy arrays.
        This method is therefore a helper for e.g. non-bitwise Random Forest learning. 
        Args: 
            tensors_list: list of torch tensors (that form the activations) to be transformed
            cuda: True if torch tensors are cuda tensors
            num_fact_bits: the number of fractional bits from the total bits (i.e. the number of bits that should be used to represent that part behind the comma)
            num_total_bits: the total number of bits wanted 
            activations_output_mode: 0 if net activations are processed, 1 if net ouputs are processed
        Returns: 
            list of tensors transformed to numpy arrays for further processing (NOTE: the values in the arrays are ints )
    '''
    l = []
    for a in tensors_list: 
        if cuda: 
            a = a.clone().cpu().detach().numpy()
        else: 
            a = a.clone().detach().numpy()
        
        a = np.array(a, dtype=np.float)
        x = np.zeros(a.shape, dtype=np.uint8)
        for idx in np.ndindex(*a.shape): # iterate over all possible indices and apply the quantization
            x[idx] = convert_float_to_quantized_int_repr(float(a[idx]), num_fract_bits, num_total_bits) # getting the int representation
        l.append(x)
    return l

def blif_file_to_graphviz_visualization(full_blif_file_path, store_folder_path, store_file_name): 
    '''
        Takes a blif file and reads it to create a visualization of the nodes that are connected with each other 
        Args: 
            full_blif_file_path: path to the blif file (including ending .blif)
            store_folder_path: folder in which to save the visualization (including last slash)
            store_file_name: the file name under which to save the visualization (without last ending)
    '''
    # directed graph
    graph = pydot.Dot(graph_type='digraph')

    # read the file
    lines = []
    with open(full_blif_file_path, "r") as f: 
        lines = f.readlines()
    
    nodes = {}

    # go through the lines and create the graphviz nodes if not yet created
    for l in lines: 
        if '.inputs' in l or '.outputs' in l or '.names' in l: 
            names = l.replace('\n', '').split(' ')
            for i in range(1, len(names)): 
                n = names[i]
                if n not in nodes.keys(): 
                    node = pydot.Node(n, color='blue4', shape='circle')
                    nodes[n] = node
                    graph.add_node(node)

    # go through the lines and create the edges
    for l in lines: 
        if '.names' in l: 
            names = l.replace('\n', '').split(' ')
            out_node = names[-1]
            for i in range(1, len(names)-1): 
                n = names[i]
                graph.add_edge(pydot.Edge(nodes[n], nodes[out_node], label='', arrowsize=0.5))

    graph.write_png('{}{}.png'.format(store_folder_path, store_file_name))


def layer_connection_graphviz_visualization(layer_connections_list, store_folder_path, store_file_name, number_nodes_list=None): 
    '''
        Takes a list of connected layers and draws a graph from it that shows the connections between the layers. 
        Args: 
            layer_connections_list: list with sequence of two ints that make up the indices of forward connected layers
            store_folder_path: folder in which to save the visualization (including last slash)
            store_file_name: the file name under which to save the visualization (without last ending)
            number_nodes_list: list with number of nodes per layer for additional information in visualization (layer_idx=list_idx) (optional, otherwise None)
    '''
    # directed graph
    graph = pydot.Dot(graph_type='digraph')

    # create the nodes: 
    nodes = {}
    for seq in layer_connections_list: 
        for idx in seq: 
            if idx not in nodes.keys(): 
                if number_nodes_list: 
                    name = 'l{}\n[{} nodes]'.format(idx, number_nodes_list[idx])
                else: 
                    name = 'l{}'.format(idx)
                node = pydot.Node(name, color='blue4', shape='box')
                nodes[idx] = node
                graph.add_node(node)

    # go through the connection sequences and create the edges
    for seq in layer_connections_list: 
        idx1 , idx2 = seq
        graph.add_edge(pydot.Edge(nodes[idx1], nodes[idx2], label='', arrowsize=0.5))

    graph.write_png('{}{}.png'.format(store_folder_path, store_file_name))


def collect_activations_and_outputs_blockwise(accumulated_truthtables, activations_to_process, outs_to_process, layer_connections_dict): 
    '''
        For block-wise processing (i.e. multiple combinations of activation relationships interdependently).
        This method concatenates newly collected activations and outputs to old ones already collected. 
        This is done in order to later calculate a truthtable from all accumulated information after one epoch (i.,e. all batches being processed).
        Args: 
            accumulated_truthtables: the accumulated truthtables by activations (inputs) and outputs in form of a dictionary (will be created if None object)
            activations_to_process: list of new activation np.arrays that need to be added
            outs_to_process: list of new output np.arrays that need to be added
            layer_connections_dict: a dictionary with key = number of layer (all except for 0) and value = list of incoming layers (from BNN)
        Returns: 
            dictionary with values = sequences of accumulated activations and accumulated outs with additional new information
    '''

    # NOTE: this method is where the logic block identifiers will be created for the first time and where it will be decided which layer relationships are part of the logic 
    # this can also already handle skip-connections 

    # make activations a list that also includes the last layer activations for the processing in the following for-loop
    activations_to_process.append(outs_to_process[0]) # outs only contains one matrix in the list, therefore choose index 0 to get this array

    truth_table_dict = {} 

    # iterate over all established network connections and form the logic blocks accordingly 
    for output_layer_number, input_layer_connections in layer_connections_dict.items(): 
            
        input_columns = None # the columns that form the input side of the truth table (will be concatenated)
        input_nodes_names = [] # the names of the input variables (layer nodes)
        
        for input_layer_number in sorted(input_layer_connections): # iterate over all skip connections and feed-forward connections
            if input_columns is None: 
                input_columns = activations_to_process[input_layer_number]
            else: 
                input_columns = np.concatenate((input_columns, activations_to_process[input_layer_number]), axis=1) # column-wise concatenation of the input side

            for in_column_idx in range(activations_to_process[input_layer_number].shape[1]): # creating the node-IDs
                input_nodes_names.append('l{}n{}'.format(input_layer_number, in_column_idx))

        for column_idx in range(activations_to_process[output_layer_number].shape[1]): #iterate over columns of output layer activation, i.e. output nodes
            out_column = activations_to_process[output_layer_number][:, column_idx] # define the output side of the truthtable
            # name the truth_table_dict entry after the name of the output node
            truth_table_dict['l{}n{}'.format(output_layer_number, column_idx)] = (input_nodes_names, input_columns, out_column) # fill everything into dict

    # at this point the keys of truth_table_dict are the node identifiers of ouputting nodes that have a connection to previous nodes and output sth
    # i.e. the keys are the node IDs of nodes that form the output side of a truth table 

    # first time collecting the truthtables
    if accumulated_truthtables is None: 
        return truth_table_dict

    # concatenate information from previous batches to information from new batches
    else: 
        for k in truth_table_dict.keys():
            nodes_names, acc_ins, acc_out = accumulated_truthtables[k]
            new_node_names, new_ins, new_outs = truth_table_dict[k]
            acc_ins = np.concatenate((acc_ins, new_ins), axis=0)
            acc_out = np.concatenate((acc_out, new_outs), axis=0)
            accumulated_truthtables[k] = (new_node_names, acc_ins, acc_out)
        return accumulated_truthtables


def class_balance_metric(labels_array):
    '''
        This method computes a measure for how balanced a data set is in terms of the classes, based on the Shannon Entropy. 
        NOTE: adapted, but taken from here: https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
        Args: 
            labels_array: numpy array of labels 
        Returns: 
            number between 0 and 1: 1 for balanced data, 0 for unbalanced data 
    '''

    unique, counts = np.unique(labels_array, return_counts=True)
    num_observations = len(labels_array)
    num_classes = len(unique)

    H = -sum([ (counts[i]/num_observations) * np.log((counts[i]/num_observations)) for i in range(num_classes)]) #shannon entropy
    return H/np.log(num_classes)


def compute_gcd(x, y):
    '''
        This method computes the greated common divisor (GCD) of two positive integer numbers.
        Args: 
            x: positive integer number 
            y: positive integer number 
        Returns: 
            GCD as positive integer number 
    '''
    assert isinstance(x, int) and isinstance(y, int) and x > 0 and y>0
    while(y):
        x, y = y, x % y
    return x


def compute_lcm(x, y):
    '''
        This method computes the least common multiple (LCM) of two positive integer numbers.
        The LCM of two numbers is the smallest positive integer that is perfectly divisible by the two given numbers.
        NOTE: for efficiency, making use of x * y = LCM * GCD.
        Args: 
            x: positive integer number 
            y: positive integer number 
        Returns: 
            LCM as positive integer number 
    '''
    assert isinstance(x, int) and isinstance(y, int) and x > 0 and y>0
    lcm = (x*y)//compute_gcd(x,y)
    return lcm


def mkdir(path):
    '''
        Checks for existence of a path and creates it if necessary: 
        Args: 
            path: (string) path to folder
    '''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError as e:
            if not os.path.exists(path):
                raise

def write_features_to_logicnet_data_files(features, num_bits, folder_path, file_name): 
    '''
        This method writes the np.features array of activations (input side of the truth tables) to a .data-file.
        This is done in a way that it fits to the special binary data format that LogicNet needs to be trained. 
        Args: 
            features: the array of features 
            num_bits: the number of bits per data field (according to activation quantization)
            folder_path: the path to the folder which to store the .data file in (including last slash)
            file_name: the name of the file to store the data in (without ending)
        Returns: 
            the final full path of the created file 
    '''
    # TODO: add assertions!!! 

    byteorder = 'little'
    signed = False
    filepath = folder_path + file_name + '.data'

    with open(filepath, "wb") as f:
        # start writing the preliminary information

        # num of dimensions in the following [as 4 byte-array]
        Num = 3 
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing dimensions: byte size is unequal to 4.'.format(file_name)

        # number of truth table rows (observations) [as 4 byte-array]
        Num = features.shape[0]
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing features.shape[0]: byte size is unequal to 4.'.format(file_name)

        # number of truth table columns (variables) [as 4 byte-array]
        Num = features.shape[1] 
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing features.shape[1]: byte size is unequal to 4.'.format(file_name)

        # number of bits [as 4 byte-array] 
        Num = num_bits 
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing num_bits: byte size is unequal to 4.'.format(file_name)
        
        write_data_field_to_logicnet_data_files(features, f)

    return filepath


def write_labels_to_logicnet_data_files(labels, num_bits, folder_path, file_name): 
    '''
        This method writes the np.labels array of activations (output side of the truth tables) to a .data-file.
        This is done in a way that it fits to the special binary data format that LogicNet needs to be trained. 
        Args: 
            labels: the array of labels 
            num_bits: the number of bits per data field (according to activation quantization)
            folder_path: the path to the folder which to store the .data file in (including last slash)
            file_name: the name of the file to store the data in (without ending)
        Returns: 
            the final full path of the created file 
    '''
    # TODO: add assertions!!! 

    byteorder = 'little'
    signed = False
    filepath = folder_path + file_name + '.data'

    with open(filepath, "wb") as f:
        
        # num of dimensions in the following [as 4 byte-array]
        Num = 2 
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing dimensions: byte size is unequal to 4.'.format(file_name)

        # number of truth table rows (observations) [as 4 byte-array]
        Num = labels.shape[0]
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing labels.shape[0]: byte size is unequal to 4.'.format(file_name)

        # number of bits [as 4 byte-array] 
        Num = num_bits 
        size = f.write(Num.to_bytes(4, byteorder=byteorder, signed=signed))
        assert size == 4, '{}: Writing num_bits: byte size is unequal to 4.'.format(file_name)

        write_data_field_to_logicnet_data_files(labels, f)
    
    return filepath


def write_data_field_to_logicnet_data_files(data_field, file): 
    '''
        This method writes the actual data field (like bitarray) of activations to a .data-file.
        This is done in a way that it fits to the special binary data format that LogicNet needs to be trained. 
        Args: 
            data_field: the array containing the actual data information
            file: the file (already opened with python) to which to write to
    '''
    # TODO: add assertions!!! 

    data_field = data_field.reshape(-1)
    data_field = np.array(data_field, dtype=np.str)
    # convert the 1's and 0's (i.e. elements) in the data field array into a long bit string
    bit_string = ''
    for x in data_field: 
        bit_string += x

    bin_array = array("B")
    # pad split at the end, so that the last data bits make up a byte 
    bit_string = bit_string + "0" * (len(bit_string)%8)
    # split the bitstring into blocks of bitstrings with 8 elements (due to upcoming byte conversion)
    splits = [bit_string[x:x + 8] for x in range(0, len(bit_string), 8)]

    # final dumping
    for byte in splits:
        bin_array.append(int(byte, 2))
    file.write(bytes(bin_array))


def write_logicnet_flist(features_path, labels_path, folder_path, file_name):
    '''
        A method that writes the flist file that LogicNet needs as argument to access the .data files with features and labels. 
        NOTE: LogicNet actually accepts two sets of data to be written into the flist after each other: training and validations.
        NOTE: Here we do not care about validation and just double the training set, as the validation results are anyway not needed
        Args: 
            features_path: path to the binary .data files of features 
            labels_path: path to the binary .data files of labels
            folder_path: the path to the folder which to store the .data file in (without last slash)
            file_name: the name of the file to store the data in (without ending)
        Returns: 
            full path to the created flist file
    ''' 

    if not folder_path.endswith('/'): 
        folder_path = folder_path + '/'

    filepath = folder_path + file_name + '.flist'
    with open(filepath, "w") as f:
        f.write('# file: {}.flist\n'.format(file_name))
        # write the path to the training data files 
        f.write(features_path +'\n')
        f.write(labels_path +'\n')
        # write the path to the validation data files (which are actually the training data files, see Note above) 
        f.write(features_path +'\n')
        f.write(labels_path +'\n')

    return filepath

def confusion_matrix_from_preds_and_labels(num_classes, predictions, labels): 
    '''
        This method derives a confusion matrix from an array of predictions and an array of labels
        Args: 
            num_classes: the number of classes 
            predictions: numpy array of predictions 
            labels: numpy array of labels

    '''
    assert type(num_classes) == int, 'Confusion Matrix Derivation: num_classes parameter must be integer.'
    assert type(predictions).__module__ == 'numpy', 'Confusion Matrix Derivation: predictions must be a numpy array.'
    assert type(labels).__module__ == 'numpy', 'Confusion Matrix Derivation: labels must be a numpy array.'
    assert predictions.shape == labels.shape, 'Confusion Matrix Derivation: shape of labels and predictions must match.'

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(predictions)):
        pred = int(predictions[i])
        true = int(labels[i])
        confusion_matrix[pred][true] += 1 
    return confusion_matrix

def num_trainable_parameters(model): 
    '''
        Args: 
            model: a pytorch model
        Returns: 
            number of parameters to be trained
    '''
    return sum(p.numel() for p in model.parameters())