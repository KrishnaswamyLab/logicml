import csv
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from logicml.code.pipeline.utils import compute_lcm, class_balance_metric
# from logicml import *

# ------------------------ AbstractDataReader -------------------------------
class AbstractDataReader: 
    def __init__(self):
        '''
            Initialization of the abstract reader. 
            NOTE: in concrete class implement the load_np_arrays() method and overwrite the upcoming self-attributes.
        '''
       
        self.args = None
       
        # the following two parameters need to be overwritten in init when implementing concrete class
        self.num_features = None
        self.num_classes = None 

        self.folder_path = None

        self.train_batches = None
        self.test_batches = None
        self.validation_batches = None

        self.train_features = None
        self.train_labels = None 
        self.test_features = None 
        self.test_labels = None
        self.validation_features = None 
        self.validation_labels = None

        self.feature_names = None
        self.num_classes = None
        self.class_labels_dict = None


    def load_np_arrays(self): 
        '''
            Loads the data numpy arrays, if available, otherwise creates the numpy arrays. 
            NOTE: enforcing overwriting because this is the part where each reader might be different
        '''

        raise NotImplementedError('Subclasses must override load_np_arrays()')

    def unison_shuffling(self, a, b):
        '''
            Args: 
                a: first array to shuffle
                b: second array to shuffle
            Returns: 
                sequence of a and b shuffled row-wise in unison
        '''
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
       

    def prepare_initial_data(self, normalize=True): 
        '''
            Loads the numpy features and labels arrays and will shuffle the instances. 
            Lastly it will distribute them to test and training data, according to the batch size and given split.
            This method fills the self.test_batches and self.train_batches lists from __init__().
            Args: 
                normalize: Boolean to set to True if the data should be min-max normalized
        '''
        features, labels = self.load_np_arrays()
        if normalize: 
            features = self.normalize_data(features)
            
            if self.test_features is not None: # can happen in the case of MNIST data split not being used from the outside 
                self.test_features = self.normalize_data(self.test_features)
        
        num_instances = len(labels)
        
        if self.test_features is not None: # can happen in the case of MNIST data split not being used from the outside 

            # create the batches for the test dataset, but based on already given test features 
            test_set_end_index = len(self.test_labels) - len(self.test_labels) % self.args.batch_size

            i = 0
            while i <= test_set_end_index - self.args.batch_size: 
                d = {
                    'features' : self.test_features[i:i+self.args.batch_size],
                    'labels' : self.test_labels[i:i+self.args.batch_size]}
                self.test_batches.append(d)
                i += self.args.batch_size

            test_set_end_index = 0 # needed for starting the validation set from beginning of the features 

        else: # normal procedure
            # create the batches for the test dataset
            test_set_end_index_max = int(num_instances * self.args.test_split)
            # at this point the test_set_end_index_max is of the maximum size we allow it to be --> test_set_end_index derived later

            # NOTE: this option makes the test data set perfectly balanced among all classes
            if self.args.balanced_test_data: 
                features_sorted = np.zeros((features.shape))
                labels_sorted = np.zeros((labels.shape))
                unique, counts = np.unique(labels, return_counts=True)

                # for very unbalanced data sets, be sure to get the minimal amount of instances that we have for the underrepresented class
                min_class_instances = np.min(counts)
                
                # checking how unbalanced the data set is and deriving actions from it: 
                # cm = class_balance_metric(labels) # 1 for balanced data, 0 for unbalanced data 
                # TODO: apply threshold like e.g. if cm < 0.95, then do the following line of code? 

                # we want to keep at least of it for training 
                min_class_instances = int((1.0-self.args.unbalanced_class_train_keep)*min_class_instances)
                
                assert min_class_instances > 0, 'DataLoader: the minimal number of instances for the most under-represented class is 0.'
                
                label_splits = {}
                start = 0
                for i in range(len(unique)): 
                    where = np.where(labels == unique[i])[0]
                    labels_sorted[start:start+counts[i]] = labels[where]
                    features_sorted[start:start+counts[i]] = features[where]
                    label_splits[unique[i]] = (start, start+counts[i])
                    start = start+counts[i]

                # now we have an array for features and labels each that are sorted according to the label occurrences

                # need to make sure that the number of points in the test data fit to the number of classes and the batch size
                # although this can mean that it might deviate from the actually defined test split percentage

                # -------------------------------------
                # NOTE: the following is the new part
                smaller = min(self.num_classes, self.args.batch_size)
                larger = max(self.num_classes, self.args.batch_size)

                # least common multiplier
                test_set_end_index_min = compute_lcm(self.num_classes, self.args.batch_size)
                num_instances_to_add = int(test_set_end_index_min / self.num_classes)

                error_msg = 'DataReader: It is impossible to find a solution for the test data creation that satisfies all the desired properties: max. split percentage, maximal number of instances per class and batch size. Adapt parameters. '
                assert test_set_end_index_min <= test_set_end_index_max, error_msg
                assert num_instances_to_add <= min_class_instances, error_msg

                test_set_end_index = test_set_end_index_min
                num_instances_per_class = num_instances_to_add

                # now working the way upwards until we found the final right test_set_end_index, that fulfills the following: 
                # - the number of instances per class is not greater than the half of the instances of the most underrepresented class 
                # - it is divisible by the batch size 
                # - it is divisible by the number of classes 
                while test_set_end_index <= test_set_end_index_max and num_instances_per_class <= min_class_instances:
                    test_set_end_index += test_set_end_index_min
                    num_instances_per_class += num_instances_to_add

                features_new = np.zeros((features.shape), dtype=np.float)
                labels_new = np.zeros((labels.shape), dtype=np.float)
                test_start = 0
                training_start = test_set_end_index
                # -------------------------------------

                # now we fill the new features and labels array from the top in a sorted with exactly the same number of observations per label
                # the rest of the arrays will be filled with the rest of the observations that will be used for the test and the training data
                
                for k, v in label_splits.items(): 
                    v_start, v_end = v
                    diff = v_end - v_start
                    
                    # test data 
                    features_new[test_start:(test_start+num_instances_per_class)] = features_sorted[v_start:(v_start+num_instances_per_class)]
                    labels_new[test_start:test_start+num_instances_per_class] = labels_sorted[v_start:v_start+num_instances_per_class]
                    # training data
                    features_new[training_start:training_start+diff-num_instances_per_class] = features_sorted[v_start+num_instances_per_class:v_end]
                    labels_new[training_start:training_start+diff-num_instances_per_class] = labels_sorted[v_start+num_instances_per_class:v_end]
                    # index updates 
                    training_start += diff-num_instances_per_class
                    test_start += num_instances_per_class

                # shuffle only the training (and validation) features and labels before re-concatenation and creation of batches
                training_features = features_new[test_set_end_index:]
                training_labels = labels_new[test_set_end_index:]
                test_features = features_new[:test_set_end_index]
                test_labels = labels_new[:test_set_end_index]
                training_features, training_labels = self.unison_shuffling(training_features, training_labels)
                
                features = np.concatenate((test_features, training_features), axis=0)
                labels = np.concatenate((test_labels, training_labels), axis=0)

            self.test_features = np.zeros((test_set_end_index, features.shape[1]))
            self.test_labels = np.zeros((test_set_end_index, 1))

            i = 0
            while i <= test_set_end_index - self.args.batch_size: 
                d = {
                    'features' : features[i:i+self.args.batch_size],
                    'labels' : labels[i:i+self.args.batch_size]}
                self.test_batches.append(d)
                self.test_features[i:i+self.args.batch_size] = features[i:i+self.args.batch_size]
                self.test_labels[i:i+self.args.batch_size] = labels[i:i+self.args.batch_size]
                i += self.args.batch_size

        # create the batches for the validation dataset
        validation_set_end_index = int((num_instances - test_set_end_index) * self.args.validation_split)
        validation_set_end_index = validation_set_end_index - validation_set_end_index % self.args.batch_size
        validation_set_end_index = validation_set_end_index + test_set_end_index

        validation_set_size = validation_set_end_index - test_set_end_index

        self.validation_features = np.zeros((validation_set_size, features.shape[1]))
        self.validation_labels = np.zeros((validation_set_size, 1))

        i = test_set_end_index
        while i <= validation_set_end_index - self.args.batch_size: 
            d = {
                'features' : features[i:i+self.args.batch_size],
                'labels' : labels[i:i+self.args.batch_size]}
            self.validation_batches.append(d)
            self.validation_features[i-test_set_end_index:i-test_set_end_index+self.args.batch_size] = features[i:i+self.args.batch_size]
            self.validation_labels[i-test_set_end_index:i-test_set_end_index+self.args.batch_size] = labels[i:i+self.args.batch_size]
            i += self.args.batch_size


        # create the batches for the training dataset
        train_set_size = num_instances - validation_set_end_index
        train_set_size -= train_set_size % self.args.batch_size
        self.train_features = np.zeros((train_set_size, features.shape[1]))
        self.train_labels = np.zeros((train_set_size, 1))

        i = validation_set_end_index
        while i <= num_instances - self.args.batch_size: 
            d = {
                'features' : features[i:i+self.args.batch_size],
                'labels' : labels[i:i+self.args.batch_size]}
            self.train_batches.append(d)
            self.train_features[i-validation_set_end_index:i-validation_set_end_index+self.args.batch_size] = features[i:i+self.args.batch_size]
            self.train_labels[i-validation_set_end_index:i-validation_set_end_index+self.args.batch_size] = labels[i:i+self.args.batch_size]
            i += self.args.batch_size

        # in case the validation split was 0 % : use the test data also as validation data, because validation data can't be zero
        if self.validation_labels.shape[0] == 0: 
            self.validation_labels = self.test_labels
            self.validation_features = self.test_features
            self.validation_batches = self.test_batches

        unique, counts = np.unique(self.test_labels, return_counts=True)


    def normalize_data(self, features): 
        '''
            Normalizes the data by the normalization mode set in argparser settings.
            
            Data Normalization Mode: 
            1: Z-Score Normalization (Standard)
            2: Min-Max-Scaling
            3: Max-Abs-Scaling
            4: Robust-Scaling 
            5: Power-Transforming
            6. Quantile-Transforming
            7. Independent Normalization 
            else: no normalization, just returning features

            See examples here: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py 

            Args: 
                features: the numpy array to normalize along the columns
            Returns: 
                (normalized or untouched) features array
        '''
        if self.args.data_normalization_mode not in range(1, 8): 
            return features

        switcher = {
            1 : StandardScaler,
            2 : MinMaxScaler,
            3 : MaxAbsScaler,
            4 : RobustScaler,
            5 : PowerTransformer, 
            6 : QuantileTransformer, 
            7 : Normalizer
        } 

        scaler = switcher[self.args.data_normalization_mode]()
        # scaler.fit(features) scaler.transform(features)
        features = scaler.fit_transform(features)
        return features


    def shuffle_and_tie_new_batches(self, test_data=False): 
        '''
            Keeps the division of training and testing data. 
            But for each shuffles the instances and forms new batches for next training epoch.
            This method overwrites self.test_batches and self.train_batches from __init__().
            Args: 
                test_data: Boolean to set to True if the test_data and validation_data should be shuffled to new batches, otherwise will be applied to training data
        '''
        
        if test_data: 
            # apply to test data 
            self.test_features, self.test_labels = self.unison_shuffling(self.test_features, self.test_labels)
            # create new batches for the test dataset
            self.test_batches = []
            i = 0
            while i <= len(self.test_labels) - self.args.batch_size: 
                d = {
                    'features' : self.test_features[i:i+self.args.batch_size],
                    'labels' : self.test_labels[i:i+self.args.batch_size]}
                self.test_batches.append(d)
                i += self.args.batch_size
            
            # apply to validation data 
            self.validation_features, self.validation_labels = self.unison_shuffling(self.validation_features, self.validation_labels)
            # create new batches for the test dataset
            self.validation_batches = []
            i = 0
            while i <= len(self.validation_labels) - self.args.batch_size: 
                d = {
                    'features' : self.validation_features[i:i+self.args.batch_size],
                    'labels' : self.validation_labels[i:i+self.args.batch_size]}
                self.validation_batches.append(d)
                i += self.args.batch_size

        else: 
            # shuffle training instances
            self.train_features, self.train_labels = self.unison_shuffling(self.train_features, self.train_labels)
            # create new batches for the test dataset
            self.train_batches = []
            i = 0
            while i <= len(self.train_labels) - self.args.batch_size: 
                d = {
                    'features' : self.train_features[i:i+self.args.batch_size],
                    'labels' : self.train_labels[i:i+self.args.batch_size]}
                self.train_batches.append(d)
                i += self.args.batch_size


# ------------------------ Coimbra Reader -------------------------------

class CoimbraReader(AbstractDataReader): 
    def __init__(self, args, folder_path=None, load_from_snapshot=False):
        '''
            Args: 
                args: the argparser arguments
                folder_path: folder in which the csv file is located when data is not located in this git
                load_from_snapshot: boolean to set to True when the data reader is loaded from a snapshot within the Trainer class 
        '''

        super(AbstractDataReader, self).__init__()

        self.args = args

        if folder_path is None: 
            direct = os.path.dirname(os.path.abspath(__file__))
            direct = os.path.dirname(direct)
            self.folder_path = os.path.dirname(direct) + '/data/breast-cancer-coimbra/'
        else: 
            self.folder_path = folder_path

        self.train_batches = [] # will contain training batches with dictionaries {'features': np.array, 'labels': np.array}
        self.test_batches = [] # will contain test batches with dictionaries {'features': np.array, 'labels': np.array}
        self.validation_batches = [] # will contain validation batches with dictionaries {'features': np.array, 'labels': np.array}

        # the following will be replaces within self.prepare_data() and stores features and labels used for training and testing (for shuffling later)
        self.train_features = None
        self.train_labels = None 
        self.test_features = None 
        self.test_labels = None
        self.validation_features = None
        self.validation_labels = None 
        self.num_features = None 

        self.feature_names = None # list that will be filled when loading the data 
        
        # something that we know about the data 
        self.num_classes = 2 
        self.class_labels_dict = {0: 'class0', 1: 'class1'}

        if not load_from_snapshot: 
            # will fill the train_batches and test_batches list with corresponding dictionaries
            self.prepare_initial_data()
            self.num_features = self.train_features.shape[1]

    
    def load_np_arrays(self, dont_store=False, shuffle=True): 
        '''
            Loads the data numpy arrays, if available, otherwise creates the numpy arrays. 
            Args: 
                dont_store: True if numpy arrays should not be stored after creation for reloading in next round
                shuffle: True if features and labels should be shuffled row-wise in unison before being returned
            Returns: 
                sequence of numpy arrays for features and labels
        '''

        if os.path.isfile(self.folder_path + 'features.npy') and os.path.isfile(self.folder_path + 'labels.npy') and os.path.isfile(self.folder_path + 'feature_names.npy') and not self.args.enforce_new_data_creation:
            features = np.load(self.folder_path + 'features.npy')
            labels = np.load(self.folder_path + 'labels.npy')
            feature_names = np.load(self.folder_path + 'feature_names.npy')
            self.feature_names = features_names.tolist()
        
        else: 
            features = []
            labels = []
            features_names =[]

            with open(self.folder_path + 'dataR2.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                line_count = 0
                for row in reader:
                    if line_count == 0:
                        features_names = row 
                    else: 
                        features.append(row[:-1])
                        labels.append(row[-1])
                    line_count += 1 
            
            features = np.array(features, dtype=np.float)
            labels = np.array(labels, dtype=np.float)
            labels[np.where(labels == 1.0)] = 0.0
            labels[np.where(labels == 2.0)] = 1.0
            features_names = np.array(features_names, dtype=str)

            self.feature_names = features_names.tolist()
            
            # reshaping to size (n, 1) instead of (n,) because pytorch wants it like that
            labels_reshaped = np.ones((labels.shape[0], 1), dtype=np.float) # pylint: disable=E1136
            labels_reshaped[:, 0] = labels[:]
            labels = labels_reshaped

            if not dont_store: 
                np.save(self.folder_path + 'features', features)
                np.save(self.folder_path + 'features_names', features_names)
                np.save(self.folder_path + 'labels', labels)
            
        if shuffle: 
            features, labels = self.unison_shuffling(features, labels)

        return features, labels

    def __repr__(self): 
        return 'CoimbraReader'


# ------------------------ MNIST Reader -------------------------------

class MNISTReader(AbstractDataReader): 
    def __init__(self, args, load_from_snapshot=False):
        '''
            Args: 
                args: the argparser arguments
                load_from_snapshot: boolean to set to True when the data reader is loaded from a snapshot within the Trainer class 
        '''

        super(AbstractDataReader, self).__init__()

        self.args = args

        self.folder_path = None

        self.train_batches = [] # will contain training batches with dictionaries {'features': np.array, 'labels': np.array}
        self.test_batches = [] # will contain test batches with dictionaries {'features': np.array, 'labels': np.array}
        self.validation_batches = [] # will contain validation batches with dictionaries {'features': np.array, 'labels': np.array}

        # the following will be replaces within self.prepare_data() and stores features and labels used for training and testing (for shuffling later)
        self.train_features = None
        self.train_labels = None 
        self.test_features = None 
        self.test_labels = None
        self.validation_features = None
        self.validation_labels = None 
        self.num_features = None 

        self.feature_names = [] # will be filled when loading the data 
        
        # something that we know about the data 
        self.num_classes = 10
        self.class_labels_dict = {
            0: 'class0', 
            1: 'class1',
            2: 'class2', 
            3: 'class3', 
            4: 'class4', 
            5: 'class5',
            6: 'class6',
            7: 'class7',
            8: 'class8',
            9: 'class9'}

        if not load_from_snapshot: 
            # will fill the train_batches and test_batches list with corresponding dictionaries
            self.prepare_initial_data()
            self.num_features = self.train_features.shape[1]


    
    def load_np_arrays(self, shuffle=True): 
        '''
            Loads the data numpy arrays from tensorflow. 
            Args: 
                shuffle: True if features and labels should be shuffled row-wise in unison before being returned
            Returns: 
                sequence of numpy arrays for features and labels
        '''
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train_orig_shape = x_train.shape
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # writing feature names - pixel_<width>-<height>
        for i in range(28): 
            for j in range(28): 
                self.feature_names.append('pixel_w{}-h{}'.format(i, j))

        if not self.args.mnist_use_args_test_splits: # when the default MNIST train and test splits should be used, instead of the given test data set argparser split
            self.test_features = x_test
            self.test_labels = y_test.reshape((y_test.shape[0], 1)) # make it shape (n, 1) instead of (n,) - used for other operations in the framework
            # NOTE: in this case we just return the training features and labels, but did already assign the test features and labels - will be recognized in other methods 
            features = x_train
            labels = y_train
        
        else: 
            # NOTE: fusing MNIST training and test data set because using argparser option from the outside to decide for test and training splits
            # also for being consistent with other data readers
            features = np.concatenate((x_train, x_test), axis=0)
            labels = np.concatenate((y_train, y_test), axis=0)
        
        labels = labels.reshape((labels.shape[0], 1)) # make it shape (n, 1) instead of (n,) - used for other operations in the framework

        if shuffle: 
            features, labels = self.unison_shuffling(features, labels)

        return features, labels

    def __repr__(self): 
        return 'MNISTReader'