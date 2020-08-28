import os
import re
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import math
from logicml.code.pipeline.utils import convert_quantized_int_repr_to_binary_string, convert_float_to_quantized_int_repr, largest_unsigned_int
from sklearn.tree import _tree, export_text
import math


def tree_predict(tree, node, x):
    '''
        This method returns the class prediction of a node of a single estimator (i.e. decision tree) in a random forest for a given example.
        Args: 
            tree: an sklearn estimator from the random forest model (i.e. a decision tree)
            node: the node within the decision tree
            x: the observation, i.e. data point example
        Returns: 
            a single prediction at one specific node of the overall tree
    '''
    assert node >= 0
    left   = tree.children_left
    right  = tree.children_right # -1 is sentinel for none
    feats  = tree.feature # -2 is sentinel for none
    thresh = tree.threshold
    values = tree.value
  
    if feats[node] == -2: # leaf node
        assert left[node] == -1
        assert right[node] == -1
        return values[node] / values[node].sum()
    else:
        assert left[node] != -1
        assert right[node] != -1
        # note: int(threshold) since we don't think it matters, as the features are all ints anyway
        if x[feats[node]] <= int(thresh[node]):
            return tree_predict(tree, left[node], x)
        else:
            return tree_predict(tree, right[node], x)

def forest_predict(model, x, debug=False):
    '''
        This method returns the class prediction of a random forest for a given example.
        Args: 
            model: the random forest sklearn model
            x: the observation, i.e. data point example
            debug: Boolean to set to True for additional print-outs in debugging mode
        Returns: 
            the overall prediction of the random forest model
    '''
    # summing up predictions and taking argmax
    # two returns: prediction and number of examples for which this is -1 or 1
    # estimators_ array contains trees (attribute .tree)
    res = tree_predict(model.estimators_[0].tree_, 0, x)  
    for estimator in model.estimators_[1:]:
        # sum up outputs across trees
        res += tree_predict(estimator.tree_, 0, x)
  
    if debug:
        print(res.reshape(-1).astype(np.int32))

    return res.reshape(-1).argmax()


def accuracy(model, examples):
    '''
        Returns: 
            the accuracy of the random forest for each example 
    '''
    return np.array([forest_predict(model, example) for example in examples])


def generate_forest(name, features, labels, n_estimators, max_depth, validation_features=None, validation_labels=None, verification=True, inverse_weighting=True, additional_return=False):
    '''
        Method that generates a random forest and trains it on the data of activations. 
        One random forest models the relationship of one NN layer to one of the next layer's node.

        NOTE: the model needs to be trained on the already tranformed data according to the quantization scheme 
            - either already bitstring and then bit by bit training, where each bit is a float value of either 0.0 or 1.0
            - or quantized version of the data to integer representation that is then internally (by sklearn) transformed into float values

        Args:
            name:  
            features: np.array that encodes the input side of the truth table (activations of previous layer)
            labels: np.array that encodes the output side of the truth tables (activation of next layer's node)
            n_estimators: number of estimators (decision trees) in the random forest
            max_depth: maximal depth of the trees in the random forest
            validation_features: features array for validation (if None, training features are used)
            validation_labels: labels array for validation (if None, training labels are used)
            verification: Boolean to set to True if a verification should be done that the own accuracy method is the same as the sklearn one 
            inverse_weighting: Boolean to set to True if for the random fortest training, the class with lower occurrence should be upweighted to make decision trees more equal
            additional_return: Boolean to set to True if training accuracy and validation accuracy should also be returned
        Returns: 
            sequence of the random forest as sklearn model and print out message (and training accuracy and validation accuracy if additional_return=True)

    '''

    training_features = features 
    training_labels = labels
    assert (validation_features is None and validation_labels is None) or (not (validation_features is None) and not (validation_labels is None)), 'Unclear definition for random forest validation.'
    if validation_features is None and validation_labels is None: 
        validation_features = features
        validation_labels = labels
    else: 
        assert validation_features.shape[0] == validation_labels.shape[0], 'Validation features and labels shape do not match for random forest training.'

    if inverse_weighting: 
        # NOTE: usually one class is dominant, e.g. in bitwise training class 0 (i.e. label of the bit = 0)
        # this leads to a decision tree that seldomly has a final leave that predicts the less common class label 
        # to pay more attention to the under-represented class, the inverse weighting option can be used to make the decision tree more balanced
        unique, counts = np.unique(training_labels, return_counts=True)
        weights = counts / np.sum(counts)
        weights = 1 / weights
        class_weight_dict = {}
        for i in range(len(unique)): 
            class_weight_dict[unique[i]] = weights[i]
    else: 
        class_weight_dict = None

    # create a (tiny tree) random forest model and run it on the training set
    # NOTE: turn off bootstrap so that samples are taken without resampling
    # and as a result sample weights are always 1 and so inference is simpler
    m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=False, random_state=0, class_weight=class_weight_dict)
    m.fit(training_features, training_labels)

    # score on training and validation set 
    training_accuracy = m.score(training_features, training_labels)
    validation_accuracy = m.score(validation_features, validation_labels)
    msg = 'name = {}, training_accuracy = {}, validation_accuracy = {}'.format(name, training_accuracy, validation_accuracy)

    if verification: # not trying to extract model from sklearn, but instead with own procedure, therefore the option to verify that 
        # number of verification samples from training data (always use ca. 1/6)
        nverify = int(float(training_features.shape[0])/6.0)
        # then own accuracy model, traverses model and extracts predictions
        own   = accuracy(m, training_features[:nverify])
        # should be the same as the one from sklearn:
        golden = m.predict(training_features[:nverify])
        # therefore compare
        assert (own == golden).all(), 'Random Forest Verification Procedure: own accuracy procedure [{}] did not lead to same result as sklearn procedure [{}].'.format(own, golden)

    if additional_return: 
        return m, msg, training_accuracy, validation_accuracy
    return m, msg

def dump_tree(tree, node, tree_id, n_classes_y, file, variable_prefix, num_total_bits, num_fract_bits):
    '''
        Regressive helper-method to dump a random tree structure and its leaves (which can again be trees) 
        into an equivalent Verilog representation.
        Args: 
            tree: an estimator (i.e. decision tree) of the random forest model
            node: a node in the decision tree
            tree_id: the id/number of the estimator
            n_classes_y: the number of output classes
            file: the file to which the Verilog code should be written
            variable_prefix: the module name of the Verilog module to which the armax will be written
            num_total_bits: the number of total bits to which the probabilities and thresholds of the random forest are quantized  
            num_fract_bits: the number of fractional bits to which the probabilities and thresholds of the random forest are quantized  
    '''

    assert node >= 0
  
    left   = tree.children_left
    right  = tree.children_right # -1 is sentinel for none
    feats  = tree.feature # -2 is sentinel for none
    thresh = tree.threshold
    values = tree.value

    for i in range(n_classes_y):
        print('    wire [{}:0] {}_n_{}_{}_{};'.format(num_total_bits-1, variable_prefix, tree_id, node, i), file=file)
    
    if feats[node] == -2: # leaf node
        assert left[node] == -1
        assert right[node] == -1
        
        # for some reason (multi output classes?) tree.value has an extra dimension
        assert values[node].shape == (1, n_classes_y)
        class_probabilities = (values[node] / values[node].sum())[0]
        
        for i in range(n_classes_y):
            p_float = float(class_probabilities[i])
            int_repr = convert_float_to_quantized_int_repr(p_float, num_fract_bits, num_total_bits)
            bin_repr = convert_quantized_int_repr_to_binary_string(int_repr, num_total_bits)
            print('    assign {}_n_{}_{}_{} = {}\'b{}; // probability: {}'.format(variable_prefix, tree_id, node, i, num_total_bits, bin_repr, p_float), file=file)
        return
    else:
        assert left[node] != -1
        assert right[node] != -1
        
        # NOTE: we are int'ing the threshold since we don't think it matters, as the features are all ints anyway
        dump_tree(tree, left[node],  tree_id, n_classes_y, file=file, variable_prefix=variable_prefix, num_total_bits=num_total_bits, num_fract_bits=num_fract_bits)
        dump_tree(tree, right[node], tree_id, n_classes_y, file=file, variable_prefix=variable_prefix, num_total_bits=num_total_bits, num_fract_bits=num_fract_bits)

        print('    wire {}_c_{}_{};'.format(variable_prefix, tree_id, node), file=file)
        
        assert 0. <= thresh[node] 
        threshold = float(thresh[node])
        int_repr = convert_float_to_quantized_int_repr(threshold, num_fract_bits, num_total_bits)
        bin_repr = convert_quantized_int_repr_to_binary_string(int_repr, num_total_bits)
        
        print('    assign {}_c_{}_{} = {}_x{} <= {}\'b{}; // threshold: {}'.format(variable_prefix, tree_id, node, variable_prefix, feats[node], num_total_bits, bin_repr, threshold), file=file)
        
        for i in range(n_classes_y):        
            print('    assign {}_n_{}_{}_{} = {}_c_{}_{} ? {}_n_{}_{}_{} : {}_n_{}_{}_{};'.format(variable_prefix, tree_id, node, i, variable_prefix, tree_id, node, variable_prefix, tree_id, left[node], i, variable_prefix, tree_id, right[node], i), file=file)


def get_bin(x, n=0):
    '''
        Get the binary representation of a positive integer x.
        Args:
            x: (int) The integer to represent as binary string
            n: (int) Minimum number of digits. If x needs less digits in binary, the rest is filled with zeros.
        Returns:
            binary string representation (only integer bits - i.e. no fractional bits - and only treating of positive numbers)
    '''
    assert x >= 0
    return format(x, 'b').zfill(n)


def dump_argmax(num_inputs, num_total_bits, output_width, variable_prefix, f):
    '''
        Method to dump Verilog file that executes an argmax in a sequential manner. 
        Args: 
            num_inputs: the number of inputs to the argmax (corresponds to the number of classes of the random forest)
            num_total_bits: the number of bits to which the probabilities and thresholds of the random forest are quantized  
            output_width: the bit width that the argmax output should have 
            variable_prefix: the module name of the Verilog module to which the armax will be written
            f: Verilog file to write to
    '''

    assert math.log2(num_inputs) >= float(output_width)
    # do not write the input wires, as they are the output wires that do already exist from random forest tree
    # those input wires always have the name <variable_prefix>_y<index> and always have a bit width of 2*num_total_bits
    # also do not create the output wire as this is also always created in the top module
    # when using the final output wire, use the name <variable_prefix>_out with [<output_width>:0]
    
    # writing the intialization
    f.write('\n    // starting argmax of block {}\n'.format(variable_prefix))
    f.write('    // initialization with max = {}_y0\n'.format(variable_prefix))
    f.write('    wire [{}:0] {}_argmax_0;\n'.format(output_width-1, variable_prefix))
    f.write('    assign {}_argmax_0 = {}\'b{};\n'.format(variable_prefix, output_width, get_bin(0, n=output_width))) # NOTE: here different binarization because class declaration is always positive and unsigned
    f.write('    wire [{}:0] {}_max_0;\n'.format(num_total_bits*2-1, variable_prefix))
    f.write('    assign {}_max_0 = {}_y0;\n'.format(variable_prefix, variable_prefix))

    # writing the blocks of comparisons
    for i in range(1, num_inputs): 
        f.write('    // comparing max against {}_y{}\n'.format(variable_prefix, i))
        f.write('    wire {}_c_{};\n'.format(variable_prefix, i))
        f.write('    assign {}_c_{} = {}_y{} >= {}_max_{};\n'.format(variable_prefix, i, variable_prefix, i, variable_prefix, i-1))
        f.write('    wire [{}:0] {}_max_{};\n'.format(num_total_bits*2-1, variable_prefix, i))
        f.write('    assign {}_max_{} = {}_c_{} ? {}_y{} : {}_max_{};\n'.format(variable_prefix, i, variable_prefix, i, variable_prefix, i, variable_prefix, i-1))
        f.write('    wire [{}:0] {}_index_{};\n'.format(output_width-1, variable_prefix, i))
        f.write('    assign {}_index_{} = {}\'b{};\n'.format(variable_prefix, i, output_width, get_bin(i, n=output_width))) # NOTE: here different binarization because class declaration is always positive and unsigned
        f.write('    wire [{}:0] {}_argmax_{};\n'.format(output_width-1, variable_prefix, i))
        f.write('    assign {}_argmax_{} = {}_c_{} ? {}_index_{} : {}_argmax_{};\n'.format(variable_prefix, i, variable_prefix, i, variable_prefix, i, variable_prefix, i-1))
    
    # assigning the final result to the output variable
    f.write('    // final assignment of output of block {}\n'.format(variable_prefix))
    f.write('    assign {}_out = {}_argmax_{};\n'.format(variable_prefix, variable_prefix, num_inputs-1))

def random_forest_dump_verilog_bitwise(model, num_inputs, n_classes_y, num_total_bits, num_fract_bits, num_input_bits, argmax_output_bits, folder_path, module_name):
    '''
        Method to dump the random forest into verilog, including preliminary information like the actual module declaration.
        NOTE: This is a method used for bitwise training.
        Args: 
            model: the sklearn model of the random forest
            num_inputs: number of inputs that the verilog module has (i.e. number of nodes from previous NN layer)
            n_classes_y: the number of outputting classes of the random forest (which is equal to len(np.unique(labels) of the labels with which the model has been trained)
            num_total_bits: the number of total bits that the output (before the argmax (i.e. the number of bits to which the probabilties and thresholds are quantized)
            num_fract_bits: the number of fractional bits from the total bits (i.e. the number of bits that should be used to represent that part behind the comma)
            num_input_bits: the number of bits that the inputs to the random forests have 
            argmax_output_bits: the bit width of the final argmax of a random forest Verilog module
            folder_path: path to the folder to store the verilog file in (without last slash)
            module_name: name that should be given to the file, the variable prefixes and the verilog module (without any ending)
    '''

    with open(folder_path + '/' + module_name +'.v', 'w') as f:
        f.write('// Quantization Scheme - Total Bits: {} - Fractional Bits: {}\n'.format(num_total_bits, num_fract_bits))
        f.write('module {}(\n'.format(module_name))
        
        for i in range(num_inputs):
            f.write('    input wire [{}:0] {}_x_orig{}{}\n'.format(num_input_bits-1, module_name, i, ','))       
        
        f.write('    output wire [{}:0] {}_out\n'.format(argmax_output_bits-1, module_name))       
        f.write('    );\n')

        # NOTE: the inputs are brought to the same bit width as num_total bits by: 
        # - stacking zeros at least significant bits (i.e. fractional bits = 0)
        # - stacking zeros at most significant bits (i.e. int bits) except for the least signifcant bit of the int bits 
        # this is done due to need of bringing the single-bit inputs to the same quantization concepts with fractional bits 
        # and int-bits as the probabilities and thresholds will be quantized (because comparisons against them will be made)
        # i.e. this currently assumes that the random forests are trained bit by bit
        f.write('    // block {} - bringing single-bit inputs to the correct size to match the quantization scheme of thresholds and probabilities\n'.format(module_name))
        for i in range(num_inputs):
            f.write('    wire [{}:0] {}_x{};\n'.format(num_total_bits-1, module_name, i))       
            f.write('    assign {}_x{} = {} {}\'b{}, {}_x_orig{}, {}\'b{} {};\n'.format(module_name, i, '{', num_total_bits-num_fract_bits-1, (num_total_bits-num_fract_bits-1)*'0', module_name, i, num_fract_bits, num_fract_bits*'0', '}'))   

        # intermediate output wires of the tree before argmax
        f.write('    // block {} - intermediate output wires of tree before argmax\n'.format(module_name))
        for i in range(n_classes_y):
            f.write('    wire [{}:0] {}_y{};\n'.format(num_total_bits*2-1, module_name, i))    
      
        for i, estimator in enumerate(model.estimators_):
            f.write('    // block {} - dumping tree {}\n'.format(module_name, i))
            dump_tree(estimator.tree_, node=0, tree_id=i, n_classes_y=n_classes_y, file=f, variable_prefix=module_name, num_total_bits=num_total_bits, num_fract_bits=num_fract_bits)    

            for c in range(n_classes_y):
                f.write('    wire [{}:0] {}_s_{}_{};\n'.format(num_total_bits*2-1, module_name, i, c))
                f.write('    wire [{}:0] {}_e_{}_{};\n'.format(num_total_bits*2-1, module_name, i, c))
                f.write('    assign {}_e_{}_{} = {} {}\'b{}, {}_n_{}_0_{} {};\n'.format(module_name, i, c, '{', num_total_bits, convert_quantized_int_repr_to_binary_string(0, num_total_bits), module_name, i, c, '}'))
                
                if i > 0:
                    f.write('    assign {}_s_{}_{} = {}_s_{}_{} + {}_e_{}_{};\n'.format(module_name, i, c, module_name, i - 1, c, module_name, i, c))
                else:
                    f.write('    assign {}_s_{}_{} = {}_e_{}_{};\n'.format(module_name, i, c, module_name, i, c))

        for c in range(n_classes_y):
            f.write('    assign {}_y{} = {}_s_{}_{};\n'.format(module_name, c, module_name, len(model.estimators_) - 1, c))

        dump_argmax(n_classes_y, num_total_bits, argmax_output_bits, module_name, f)

        f.write('endmodule')


def random_forest_substitute_verilog_bitwise(num_inputs, num_input_bits, num_output_bits, value_to_output, folder_path, module_name):
    '''
        Writes a Verilog file that receives inputs, but doesnt use them and just outputs a constant value.
        This needs to be done as substitute for the random forest or LogicNet if for training the data label is the same for all observed input combinations. 
        NOTE: This is a method used for bitwise training.
        Args: 
            num_inputs: number of inputs that the verilog module has (i.e. number of nodes from previous NN layer)
            num_input_bits: the number of bits that the inputs to the random forests have 
            num_output_bits: the bit width of the output
            value_to_output: the int value that needs to be turned into a binary representation
            folder_path: path to the folder to store the verilog file in (without last slash)
            module_name: name that should be given to the file, the variable prefixes and the verilog module (without any ending)
    '''

    # NOTE this whole method currently assume as bit-by-bit training of random forest!!! 
    
    assert isinstance(value_to_output, int)

    with open(folder_path + '/' + module_name +'.v', 'w') as f:
        f.write('module {}(\n'.format(module_name))
        
        for i in range(num_inputs):
            f.write('    input wire [{}:0] {}_x_orig{}{}\n'.format(num_input_bits-1, module_name, i, ','))       
        
        f.write('    output wire [{}:0] {}_out\n'.format(num_output_bits-1, module_name))       
        
        f.write('    );\n')

        f.write('    // block {} - substitute for random forest\n'.format(module_name))
        f.write('    assign {}_out = {}\'b{}; // value to output: {}\n'.format(module_name, num_output_bits, convert_quantized_int_repr_to_binary_string(value_to_output, num_output_bits), value_to_output))
        f.write('endmodule')


def write_random_forest_to_txt_file(model, folder_path, file_name, feature_names=None): 
    '''
        This method goes through the estimators of the random forest model and writes the set of rules of each estimator to a text_file
        Args: 
            model: the sklearn model of the random forest
            folder_path: path to the folder to store the verilog file in (including last slash)
            file_name: name that should be given to the file (without any ending)
            feature_names: a list of names of the features that go into the random forest
    '''

    for idx, estimator in enumerate(model.estimators_): 
        s = export_text(estimator, feature_names=feature_names)
        file_path = folder_path + file_name + '_estimator{}.txt'.format(idx)
        with open(file_path, 'w') as f: 
            f.write('Decision Tree - Estimator {}:\n--------------------------\n\n'.format(idx))
            f.write(s)
            f.write('\n\nFeature Importance:\n--------------------------\n\n')
            for i, importance in enumerate(estimator.feature_importances_): 
                f.write('{}: {:.2f} %\n'.format(feature_names[i], importance*100))


def random_forest_to_verilog_direct(model, num_inputs, num_total_bits, num_fract_bits, module_name, folder_path, threshold_lower=True, direct_rf=False, direct_rf_out_width=None): 
    '''
        This method writes the static decision rules of a random forest to a Verilog module.
        NOTE: This method does a direct translation of the decision rules, not using the probabilites of each classes implicitly at each note.

        Args: 
            model: the random forest sklearn model
            num_inputs: number of inputs that the verilog module has (i.e. number of nodes from previous NN layer)
            num_total_bits: the number of total bits used for the quantization 
            num_fractional_bits: the number of fractional bits used for the quantization
            module_name: the name of the Verilog module (also used as prefix for the variables)
            folder_path: path to the folder to store the verilog file in (including last slash)
            threshold_lower: Boolean to set to True if for a float threshold the next lower integer is chosen, otherwise the next higher integer is chosen
            direct_rf: Boolean that defines if it is a direct random forest routine (True) or just a none-bitwise random forest routine from the logic pipeline (False)
            direct_rf_out_width: when direct_rf = True, direct_rf_out_width needs to be given that defines how many bits final_out should have (analog to argmax on last layer) 
        Returns: 
            the full path to the created Verilog file
    '''
    filepath = folder_path + module_name +'.v'
    with open(filepath, 'w') as f:

        # ----- Writing module declaration ---------
        f.write('// Quantization Scheme - Total Bits: {} - Fractional Bits: {}\n'.format(num_total_bits, num_fract_bits))
        f.write('module {}(\n'.format(module_name))
            
        if not direct_rf: 
            for i in range(num_inputs):
                f.write('    input signed [{}:0] {}_x{}{}\n'.format(num_total_bits-1, module_name, i, ','))  
            f.write('    output signed [{}:0] {}_out\n'.format(num_total_bits-1, module_name))       
        else: 
            assert isinstance(direct_rf_out_width, int), 'Error in writing direct random forest Verilog logic: Output Width not given.'
            for i in range(num_inputs):
                f.write('    input signed [{}:0] final_in{}{}\n'.format(num_total_bits-1, i, ',')) 
            f.write('    output [{}:0] final_out\n'.format(direct_rf_out_width-1))     
        f.write('    );\n')

        if direct_rf: # additional wires needed, because the logic creation uses them 
            f.write('\n')
            for i in range(num_inputs):
                    f.write('    wire signed [{}:0] {}_x{};\n'.format(num_total_bits-1, module_name, i))  
                    f.write('    assign {}_x{} = final_in{};\n'.format(module_name, i, i))  

            f.write('\n    wire signed [{}:0] {}_out;\n'.format(num_total_bits-1, module_name))   

        # ----- Writing each estimator ---------
        num_estimators = 0
        for tree_id, estimator in enumerate(model.estimators_):
            tree_to_verilog_direct(estimator, tree_id, num_total_bits, num_fract_bits, module_name, f, threshold_lower=threshold_lower)
            num_estimators += 1

        # ----- Writing majority vote and ending module  ---------
        majority_voting_to_verilog_direct(num_estimators, num_total_bits, module_name, f)

        # ----- Additional handling of final wire for direct random forest  ---------
        if direct_rf: 
            # NOTE: at this point, the winner-takes-it-all logic has a number of total bits, but should have as many bits as only needed for the class declaration
            # i.e. identical to the argmax-declaration in last layer logic replacement of the full logic pipeline 
            # however, the class declaration is already correct and stands at the least significant bits, 
            # thus just cut away the upper bits and create a new final_out wire 

            f.write('    assign final_out = {}_out[{}:0];\n\n'.format(module_name, direct_rf_out_width-1))   
    
        f.write('endmodule')

    return filepath


def tree_to_verilog_direct(estimator, tree_id, num_total_bits, num_fract_bits, module_name, f, threshold_lower=True): 
    '''
        This method writes the static decision rules of a single decision tree to a Verilog module.
        NOTE: This method does a direct translation of the decision rules, not using the probabilites of each classes implicitly at each note.

        Args: 
            estimator: the decision tree estimator from the sklearn model
            tree_id: the ID of the tree
            num_total_bits: the number of total bits used for the quantization 
            num_fractional_bits: the number of fractional bits used for the quantization
            module_name: the name of the Verilog module (also used as prefix for the variables)
            f: the opened Verilog file to which it should be written
            threshold_lower: Boolean to set to True if for a float threshold the next lower integer is chosen, otherwise the next higher integer is chosen
    '''

    f.write('\n    // STARTING NEW ESTIMATOR: Tree {}\n'.format(tree_id))

    # ----- Get main information from estimator ---------

    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    features = estimator.tree_.feature
    thresholds = estimator.tree_.threshold

    # ----- Traverse the tree to gather which node is a terminal (final leaf) node ---------

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depths = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depths[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True


    # ----- Traverse the tree and write the needed declarations to Verilog (the static class predictions) ---------
    for node_id in range(n_nodes): 

        # ----- Final Leaf: Encode the class label ---------
        if is_leaves[node_id]:
            # get the values at this node
            # values is then an array of shape (1, len(estimator.classes))
            # for each class is counts how often training examples were classified as the class that corresponds to this entry
            values = estimator.tree_.value[node_id]
            
            # now derive the probabilities
            probs = values / np.sum(values)
            probs = probs[0]

            # now get the class label that has the highest probability and encode this statically
            argmax = np.argmax(values)

            # NOTE: the class label is a float with .0 ending that is only a float because sklearn stores it like this 
            # NOTE: it was however before hand derived as integer representation according to the quantization scheme  
            class_label = int(estimator.classes_[argmax])
            assert class_label <= largest_unsigned_int(num_total_bits)
            probability = float(probs[argmax])

            # write the Verilog parts 
            f.write('\n    // Tree {}: declarations for terminal leaf with node_id: {}\n'.format(tree_id, node_id))
            f.write('    wire signed [{}:0] {}_t{}_n{}_pred;\n'.format(num_total_bits-1, module_name, tree_id, node_id))
            f.write('    wire signed [{}:0] {}_t{}_n{}_prob;\n'.format(num_total_bits-1, module_name, tree_id, node_id))
            bin_repr = convert_quantized_int_repr_to_binary_string(class_label, num_total_bits)
            f.write('    assign {}_t{}_n{}_pred = {}\'b{}; // class: {}\n'.format(module_name, tree_id, node_id, num_total_bits, bin_repr, class_label))
            int_repr = convert_float_to_quantized_int_repr(probability, num_fract_bits, num_total_bits)
            bin_repr = convert_quantized_int_repr_to_binary_string(int_repr, num_total_bits)
            f.write('    assign {}_t{}_n{}_prob = {}\'b{}; // probability: {:.4f}\n'.format(module_name, tree_id, node_id, num_total_bits, bin_repr, probability))


    # ----- Traverse the tree and write the decision rules to Verilog ---------
    # NOTE: we traverse the tree backwards and propagate the the final class predictions from the leaf nodes up to the root node

    for node_id in reversed(range(n_nodes)): 

        # ----- Intermediate node - get threshold and compare to proceed to next node ---------
        if not is_leaves[node_id]:
            node_depth = node_depths[node_id] # the depth at which the node is
            left_node = children_left[node_id] # the ID of the next node for <= comparison being True
            right_node = children_right[node_id] # the ID of the next node for <= comparison being False
            threshold = float(thresholds[node_id]) # the threshold for the comparison and decision
            feature_to_compare = features[node_id] # is the index of the feature under comparison

            assert feature_to_compare >= 0 # feature_to_compare is only negative for leaf nodes as replacement for None, should therefore actually not be needed to compare here
            assert right_node > 0 # right_node is only negative for leaf nodes as replacement for None, should therefore actually not be compared here
            assert left_node > 0 # left_node is only negative for leaf nodes as replacement for None, should therefore actually not be compared here

            # if feature_to_compare < = threshold, take class prediction from the left node at next depth stage
            # else: take class prediction from right_node 

            f.write('\n    // Tree {}: decision at leaf with node_id: {}\n'.format(tree_id, node_id))
            f.write('    wire {}_t{}_n{}_c;\n'.format(module_name, tree_id, node_id))
            
            # NOTE: Some explanations on the following threshold conversion
            # Recall: 
            # - we convert activations from NN to quantization scheme to int 
            # - then we train random forest on int repr, that is internally transformed to float (by sklearn)
            # - thresholds are thus basically in this integer range, but with additional fractions
            # Thus: how to convert this back into quantization scheme? 
            # - somehow convert the threshold back into the actual float value range from which the activations originated ? 
            # - then reconvert this new float value into int and bin representation? 
            # - this is likely to result in approximation errors that we propagate by reversing the quantization and then quantizing again
            # Thus: current solution:  
            # - we always compare with <= --> we can choose a slightly smaller lower bound that maps to the same binary string representation
            # - therefore we just int away the additional fraction added by sklearn, i.e. take the next lower integer
            # - or should we add +1 to not take the next lower integer, but the next upper integer, e.g. make 67.5 not be 67, but 68 instead? --> this is an outside option
            
            int_repr = int(threshold)
            if not threshold_lower: # in this case, the next higher integer is chosen
                int_repr += 1
            # perform a clipping:
            largest = largest_unsigned_int(num_total_bits) 
            if int_repr > largest: 
                int_repr = largest
            # assert int_repr <= largest_unsigned_int(num_total_bits)
            bin_repr = convert_quantized_int_repr_to_binary_string(int_repr, num_total_bits)
            f.write('    assign {}_t{}_n{}_c = {}_x{} <= {}\'b{}; // feature {} <= {:.4f} ? \n'.format(module_name, tree_id, node_id, module_name, feature_to_compare, num_total_bits, bin_repr, feature_to_compare, threshold))
            f.write('    wire signed [{}:0] {}_t{}_n{}_pred;\n'.format(num_total_bits-1, module_name, tree_id, node_id))
            f.write('    wire signed [{}:0] {}_t{}_n{}_prob;\n'.format(num_total_bits-1, module_name, tree_id, node_id))
            f.write('    assign {}_t{}_n{}_pred = {}_t{}_n{}_c ? {}_t{}_n{}_pred : {}_t{}_n{}_pred;\n'.format(module_name, tree_id, node_id, module_name, tree_id, node_id, module_name, tree_id, left_node, module_name, tree_id, right_node))
            f.write('    assign {}_t{}_n{}_prob = {}_t{}_n{}_c ? {}_t{}_n{}_prob : {}_t{}_n{}_prob;\n'.format(module_name, tree_id, node_id, module_name, tree_id, node_id, module_name, tree_id, left_node, module_name, tree_id, right_node))

def majority_voting_to_verilog_direct(num_estimators, num_total_bits, module_name, f): 
    '''
        This method writes the final ending of the random forest, including the majority voting and endmodule declaration.
        NOTE: This method does a direct translation of the decision rules, not using the probabilites of each classes implicitly at each note.

        Args: 
            num_estimators: the number of decision trees in the sklearn model
            num_total_bits: the number of total bits used for the quantization 
            module_name: the name of the Verilog module (also used as prefix for the variables)
            f: the opened Verilog file to which it should be written
    '''

    assert num_estimators > 0, 'Number of decision trees cannot be zero to write the final declaration of the random forest logic.'
    
    if num_estimators > 1: # multiple decision trees
        f.write('\n    // STARTING MAJORITY VOTING OF RANDOM FOREST \n\n')
        
        for i in range(1, num_estimators): 
            f.write('    wire {}_majority_vote{}_c;\n'.format(module_name, i-1))
            f.write('    wire signed [{}:0] {}_majority_vote{}_pred;\n'.format(num_total_bits-1, module_name, i-1))
            if i == 1: 
                if i != num_estimators-1: # meaning there is no next cycle where this is needed
                    f.write('    wire signed [{}:0] {}_majority_vote{}_prob;\n'.format(num_total_bits-1, module_name, i-1))
                f.write('    assign {}_majority_vote{}_c = {}_t{}_n0_prob >= {}_t{}_n0_prob; // comparison of tree {} and tree {}\n'.format(module_name, i-1, module_name, i, module_name, i-1, i, i-1))
                f.write('    assign {}_majority_vote{}_pred = {}_majority_vote{}_c ? {}_t{}_n0_pred : {}_t{}_n0_pred;\n'.format(module_name, i-1, module_name, i-1, module_name, i, module_name, i-1))
                if i != num_estimators-1: # meaning there is no next cycle where this is needed
                    f.write('    assign {}_majority_vote{}_prob = {}_majority_vote{}_c ? {}_t{}_n0_prob : {}_t{}_n0_prob;\n'.format(module_name, i-1, module_name, i-1, module_name, i, module_name, i-1))
            else: 
                if i != num_estimators-1: # meaning there is no next cycle where this is needed
                    f.write('    wire signed [{}:0] {}_majority_vote{}_prob;\n'.format(num_total_bits-1, module_name, i-1))
                f.write('    assign {}_majority_vote{}_c = {}_t{}_n0_prob >= {}_majority_vote{}_prob; // comparison of tree {} and tree majority vote {}\n'.format(module_name, i-1, module_name, i, module_name, i-2, i, i-2))
                f.write('    assign {}_majority_vote{}_pred = {}_majority_vote{}_c ? {}_t{}_n0_pred : {}_majority_vote{}_pred;\n'.format(module_name, i-1, module_name, i-1, module_name, i, module_name, i-2))
                if i != num_estimators-1: # meaning there is no next cycle where this is needed
                    f.write('    assign {}_majority_vote{}_prob = {}_majority_vote{}_c ? {}_t{}_n0_prob : {}_majority_vote{}_prob;\n'.format(module_name, i-1, module_name, i-1, module_name, i, module_name, i-2))

        f.write('\n    // FINAL ASSIGNMENT \n')
        f.write('    assign {}_out = {}_majority_vote{}_pred;\n'.format(module_name, module_name, num_estimators-2))

    else: # just one single decision tree 
        f.write('\n    // FINAL ASSIGNMENT \n')
        f.write('    assign {}_out = {}_t0_n0_pred;\n'.format(module_name, module_name))
