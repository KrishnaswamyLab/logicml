import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg")
import collections
import os

# ---------------- Class: ColorScheme ----------------

class ColorScheme:
    '''
        This class defines a color scheme.
    '''

    def __init__(self, normalise_range=False):
        self.colours_RGB = {'blue': {100: (0, 84, 159), 75: (64, 127, 183), 50: (142, 186, 229), 25: (199, 221, 242), 10: (232, 214, 250)},
                            'black': {100: (0, 0, 0), 75: (100, 101, 103), 50: (156, 158, 159), 25: (207, 209, 210), 10: (236, 237, 237)},
                            'magenta': {100: (227, 0, 102), 75: (233, 96, 136), 50: (241, 158, 177), 25: (249, 210, 218), 10: (253, 238, 240)},
                            'yellow': {100: (255, 237, 0), 75: (255, 240, 85), 50: (255, 245, 155), 25: (255, 250, 209), 10: (255, 253, 238)},
                            'petrol': {100: (0, 97, 101), 75: (45, 127, 131), 50: (125, 164, 167), 25: (191, 208, 209), 10: (230, 236, 236)},
                            'turquoise': {100: (0, 152, 161), 75: (0, 177, 183), 50: (137, 204, 207), 25: (202, 231, 231), 10: (235, 246, 246)},
                            'green': {100: (87, 171, 39), 75: (141, 192, 96), 50: (184, 214, 152), 25: (221, 235, 206), 10: (242, 247, 236)},
                            'may green': {100: (189, 205, 0), 75: (208, 217, 92), 50: (224, 230, 154), 25: (240, 243, 208), 10: (249, 250, 23)},
                            'orange': {100: (246, 168, 0), 75: (250, 190, 80), 50: (253, 212, 143), 25: (254, 234, 201), 10: (255, 247, 234)},
                            'red': {100: (204, 7, 30), 75: (216, 92, 65), 50: (230, 150, 121), 25: (243, 205, 187), 10: (250, 235, 227)},
                            'bordeaux': {100: (161, 16, 53), 75: (182, 82, 86), 50: (205, 139, 135), 25: (229, 197, 192), 10: (245, 232, 229)},
                            'violet': {100: (97, 33, 88), 75: (131, 78, 117), 50: (168, 133, 158), 25: (210, 192, 205), 10: (237, 229, 234)},
                            'purple': {100: (122, 111, 172), 75: (155, 145, 193), 50: (188, 181, 215), 25: (222, 218, 235), 10: (242, 240, 247)},
                           }
        self.normalise_range = normalise_range

    def __call__(self, colour_name, colour_opacity=100):
        if self.normalise_range:
            return [c / 255 for c in self.colours_RGB[colour_name][colour_opacity]]
        else:
            return self.colours_RGB[colour_name][colour_opacity]


# ---------------- Class: TensorboardParser ----------------

class TensorboardParser:
    '''
        This class provides methods for parsing the tensorboard logs.
    '''

    def __init__(self):
        pass

    def give_logfile_paths(self, folder_path, sub_folder_names): 
        '''
            Gives the paths to tensorboard files. 
            Args: 
                folder_path: the folder to where the tensorboard subfolders are located
                sub_folder_names: names of the subfolders with tensorboard events that should be taken into account
            Return: 
                dictionary with subfolder_names as keys and the corresponding path to tensorboard file as values
        '''
        d = {} 
        for s in sub_folder_names: 
            c = folder_path + '/' + s
            f = c + '/' + os.listdir(c)[0]
            d[s] = f
        return d

    def smoothing_fct(self, values, weight=0.5):
        '''
            Returns a moving average of the values in a list as done in original tensorboard. 
            Args: 
                values: a list of values 
                weight: the smoothing weight (between 0.0 = no smoothing and 1.0 = full smoothing) - how much emphasis is put onto previous point
            Returns: 
                smoothed values
        '''
        # TODO: extend the window of the moving average to be defined from the outside??
        weight = weight / 2

        # moving average
        last = values[0]
        smoothed = []
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed


    def read_single_scalar_tensorboard(self, inputLogFile, plot_var_identifier):
        '''
            Reads a single tensorboard file and puts the scalar values of a specifier into a dict structure for further processing. 

            Args:
                inputLogFile: the path to the tensorboard event-file
                plot_var_identifier: the identifier of the curve that we should look at

            Returns:
                dictionary with keys = 'step' and '<plot_var_identifier>', values = lists of values per key 
        '''

        # checking the conditions for import of the event accumulator 
        # Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
        eventAccumulatorImported = False
        # TF version < 1.1.0
        if (not eventAccumulatorImported):
            try:
                from tensorflow.python.summary import event_accumulator
                eventAccumulatorImported = True
            except ImportError:
                eventAccumulatorImported = False
        # TF version = 1.1.0
        if (not eventAccumulatorImported):
            try:
                from tensorflow.tensorboard.backend.event_processing import event_accumulator
                eventAccumulatorImported = True
            except ImportError:
                eventAccumulatorImported = False
        # TF version >= 1.3.0
        if (not eventAccumulatorImported):
            try:
                from tensorboard.backend.event_processing import event_accumulator
                eventAccumulatorImported = True
            except ImportError:
                eventAccumulatorImported = False
        # TF version = Unknown
        if (not eventAccumulatorImported):
            raise ImportError('Could not locate and import Tensorflow event accumulator.')

        ea = event_accumulator.EventAccumulator(inputLogFile, size_guidance={event_accumulator.SCALARS: 0}) # we only want to look at scalars
        ea.Reload() # loads events from file
        tags = ea.Tags() # all potential tags from the file (like 'scalars', 'histograms', etc.)
        scalarTags = tags['scalars'] # the tags within scalars (coming from trainer)

        assert plot_var_identifier in scalarTags, 'TensorboardParser.readTB(): Given plot_var_identifier [{}] does not occur in tensorboard event - only the following: {}.'.format(plot_var_identifier, scalarTags)

        # data structures for collecting the actual data from the events
        retd = {'step': [], plot_var_identifier : []}

        # create a scalar to be read
        vals = ea.Scalars(plot_var_identifier)

        for i in range(len(vals)):
            v = vals[i]
            retd['step'].append(v.step)
            scalarTag = ea.Scalars(plot_var_identifier)
            if len(scalarTag) > i:
                S = scalarTag[i]
                retd[plot_var_identifier].append(S.value)
            else:
                retd[plot_var_identifier].append('')

        return retd


    def plot_single_run_experiments(self, folder_path, subfolder_names_alias, plot_vars_alias, plot_vars_identifiers, save_path): 
        '''
            Makes plots based on given information. 
            NOTE: specifically designed for experiments where there was only one run. 
            Args: 
                folder_path: folder to tensorboard subfolders (without last slash)
                subfolder_names: names of the experiment names that should be taken into account
                plot_vars_alias: plots that will be created with each subfolder_names_alias as one curve in it (key has to match the key in plot_vars_identifiers)
                plot_vars_identifiers: plot identifiers from the from trainer
                save_path: path to save the plots to (without last slash)
        '''
        tb_event_files = self.give_logfile_paths(folder_path, sorted(subfolder_names_alias.keys()))
        for plot_var, v in plot_vars_alias.items(): 
            plot_name, x_label, y_label, smoothing_weight = v
            
            values = {}
            max_len = 0

            # collect the values for the curves
            for sub_folder, curve in subfolder_names_alias.items(): 
                curve_name, curve_color = curve
                retd = self.read_single_scalar_tensorboard(tb_event_files[sub_folder], plot_vars_identifiers[plot_var])
                x_vals = retd['step']
                y_vals = retd[plot_vars_identifiers[plot_var]]
                length = len(y_vals)
                if length > max_len: 
                    max_len = length
                values[curve_name] = (x_vals, y_vals, curve_color)

            # now do the actual plot 
            plt.figure(figsize=(20, 10))
            plt.title(plot_name)
            plt.xlim(0, max_len)

            for curve_name, vals in values.items(): 
                x_vals, y_vals, curve_color = vals
                # apply smoothing: 
                y_vals = self.smoothing_fct(y_vals, weight=smoothing_weight)
                # plotting the original curve 
                plt.plot(x_vals, y_vals, label=curve_name, color=ColorScheme(normalise_range=True)(curve_color, 100))
            

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            
            plt.legend()
            plt.savefig(save_path + '/{}.png'.format(plot_name))
            print('Stored plot: {}'.format(save_path + '/{}.png'.format(plot_name)))


    def plot_multi_run_experiments(self, folder_path, subfolder_names_alias, plot_vars_alias, plot_vars_identifiers, save_path, num_runs_dict): 
        '''
            Makes plots based on given information. 
            NOTE: specifically designed for experiments with multiple runs. 
            Args: 
                folder_path: folder to tensorboard subfolders (without last slash)
                subfolder_names: names of the experiment names that should be taken into account (without the '_run..' part in the end)
                plot_vars_alias: plots that will be created with each subfolder_names_alias as one curve in it (key has to match the key in plot_vars_identifiers)
                plot_vars_identifiers: plot identifiers from the from trainer
                save_path: path to save the plots to (without last slash)
                num_runs_dict: the number of runs
        '''

        subfolder_names_alias_original = subfolder_names_alias.copy()

        # identifiers of the runs
        runs_dict = {}
        # adapting the subfolder_names_alias for searching the tb event files and runs_dict
        for k, v in num_runs_dict.items(): 
            for i in range(v): 
                subfolder_names_alias[k + '_run{}'.format(i)] = subfolder_names_alias_original[k]
                if k not in runs_dict.keys(): 
                    runs_dict[k] = [k + '_run{}'.format(i)]
                else: 
                    runs_dict[k].append(k + '_run{}'.format(i))
            del subfolder_names_alias[k]

        tb_event_files = self.give_logfile_paths(folder_path, sorted(subfolder_names_alias.keys()))

        # iterating over the plots to make
        for plot_var, v in plot_vars_alias.items(): 
            plot_name, x_label, y_label, smoothing_weight, show_only_mean_curves = v
            
            values = {}
            max_len = 0

            # collect the values for all curves within that plot
            for sub_folder, curve in subfolder_names_alias.items(): 
                curve_name, curve_color = curve
                retd = self.read_single_scalar_tensorboard(tb_event_files[sub_folder], plot_vars_identifiers[plot_var])
                x_vals = retd['step']
                y_vals = retd[plot_vars_identifiers[plot_var]]
                length = len(y_vals)
                if length > max_len: 
                    max_len = length
                values[sub_folder] = (x_vals, y_vals, curve_color)

            # setup the actual plot 
            plt.figure(figsize=(20, 10))
            plt.title(plot_name)
            plt.xlim(0, max_len)

            if not show_only_mean_curves: 
                # plot all original unsmoothed curves with no label and less opacity
                for curve_name, vals in values.items(): 
                    x_vals, y_vals, curve_color = vals
                    # apply smoothing: 
                    y_vals = self.smoothing_fct(y_vals, weight=smoothing_weight)
                    plt.plot(x_vals, y_vals, color=ColorScheme(normalise_range=True)(curve_color, 25))

            # calculate the smoothing vals and plot smoothed curves with label and full opacity
            for k, v in subfolder_names_alias_original.items(): 
                curve_name, curve_color = v
                x_vals = None
                mean_vals = []
                # create list with the right values array
                for run_id in runs_dict[k]: 
                    x_vals, y_vals, curve_color = values[run_id] 
                    if len(y_vals) == 300: 
                        y_vals = y_vals[:-1]
                    mean_vals.append(y_vals)

                mean_vals = np.array(mean_vals)
                # mean alues now have the shape that the columns rows are the number of runs and the columns are the the values
                mean_vals = np.mean(mean_vals, axis=0).tolist()
                # apply smoothing: 
                mean_vals = self.smoothing_fct(mean_vals, weight=smoothing_weight)
                plt.plot(x_vals, mean_vals, label=curve_name, color=ColorScheme(normalise_range=True)(curve_color, 100))

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            
            plt.legend()
            plt.savefig(save_path + '/{}.png'.format(plot_name))
            print('Stored plot: {}'.format(save_path + '/{}.png'.format(plot_name)))



# ---------------- Main ----------------

if __name__ == "__main__":

    # This script can read information from tensorboard events and create plots with mutliple (smoothed) curves with matplotlib. 
    # This can be of interest, when e.g. wanting to plot loss curves for presentations, etc.

    # NOTE: the following stuff is just an example and needs to be adapted
    # NOTE: since the corresponding tensorboard events are not part of this repro, this script will currently not work - once more: adapt it yourself! 

    # ++++++++++++++++ Stuff to leave untouched in general ++++++++++++++++ 
    
    folder_path = '/path/to/tensorboards' # NOTE: without last slash
    save_path = '/path/to/save/files/to'  # NOTE: without last slash
    
    # plot identifiers from the from trainer
    plot_vars_identifiers = {
        'Training Loss': '00_General/TrainingRunningLoss', 
        'Training Loss Detailed' : '99_Detailed/DetailedTrainingLossPerBatch', 
        'Validation Loss': '00_General/ValidationRunningLoss', 
        'Validation Loss Detailed': '99_Detailed/DetailedValidationLossPerBatch', 
        'Validation Accuracy': '00_General/ValidationRunningAccuracy'}


    # ++++++++++++++++ Single Run (No Smoothing) Plot Setup  ++++++++++++++++ 
    # the names of subfolder experiments that should be taken into account and the label with shich the should occurr in the plot and their RWTH color
    subfolder_names_alias_no_smoothing = {
        '0norm_adam_bce' : ('0norm_adam_bce', 'black'), 
        '1norm_adam_bce' : ('1norm_adam_bce', 'blue'), 
        '2norm_adam_bce' : ('2norm_adam_bce', 'magenta'),
        '3norm_adam_bce' : ('3norm_adam_bce', 'green'),
        '4norm_adam_bce' : ('4norm_adam_bce', 'red'), 
        '5norm_adam_bce' : ('5norm_adam_bce', 'yellow'), 
        '6norm_adam_bce' : ('6norm_adam_bce', 'petrol'),
        '7norm_adam_bce' : ('7norm_adam_bce', 'orange')} 

    # the names of the plots that will be created with each subfolder_names_alias as one curve in it
    # key has to match the key in plot_vars_identifiers
    # value has to be a sequence of plot's name, xlabel, ylabel and smoothing value
    plot_vars_alias_no_smoothing = {
        'Training Loss' : ('Single Experiments - Training: Running Loss (Smoothing 1.0)', 'Epoch', 'BCE Loss', 1.0), 
        'Validation Loss' : ('Single Experiments - Validation: Running Loss (Smoothing 1.0)', 'Epoch', 'BCE Loss', 1.0),
        'Validation Accuracy' : ('Single Experiments - Validation: Accuracy (Smoothing 1.0)', 'Epoch', 'Accuracy', 1.0)
    }

    # ++++++++++++++++ Doing the plotting ++++++++++++++++ 

    tbp = TensorboardParser()
    tbp.plot_single_run_experiments(folder_path, subfolder_names_alias_no_smoothing, plot_vars_alias_no_smoothing, plot_vars_identifiers, save_path)



    # ++++++++++++++++ Multiple Runs (Smoothing) Plot Setup  ++++++++++++++++ 

    # -------------------- ADAM -------------------- 

    subfolder_names_alias_smoothing = {
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode1*no_logic' : ('Normalization Mode: 1 (Z-Score-Normalization)', 'red'), 
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode2*no_logic' : ('Normalization Mode: 2 (Min-Max-Normalization)', 'blue'), 
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode3*no_logic' : ('Normalization Mode: 3 (Max-Abs-Scaling)', 'black'), 
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode4*no_logic' : ('Normalization Mode: 4 (Robust Scaling)', 'petrol'), 
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode5*no_logic' : ('Normalization Mode: 5 (Power Transforming)', 'orange'), 
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode6*no_logic' : ('Normalization Mode: 6 (Quantile Transforming)', 'purple'),
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode7*no_logic' : ('Normalization Mode: 7 (Independent Normalization)', 'green') } 

    # dictionary with the same keys as in subfolder_names_alias that state the number of runs as values
    num_runs_dict = {
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode1*no_logic' : 3,
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode2*no_logic' : 3,
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode3*no_logic' : 3,
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode4*no_logic' : 3,
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode5*no_logic' : 3,
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode6*no_logic' : 3,
        'layer_outnodes_9_9_9_9*optimizeradam*data_normalization_mode7*no_logic' : 3
    }

    # the names of the plots that will be created with each subfolder_names_alias as one curve in it
    # key has to match the key in plot_vars_identifiers
    # value has to be a sequence of plot's name, xlabel, ylabel, smoothing value and option to only show mean curve of all runs
    plot_vars_alias_smoothing = {
        'Training Loss' : ('Training: Running Loss (Smoothing 0.5) | Hidden Nodes: [9-9-9-9], Optimizer: Adam, Runs: 3, No Logic | Comparing Normalization', 'Epoch', 'BCE Loss', 0.5, True), 
        'Validation Loss' : ('Validation: Running Loss (Smoothing 0.5) | Hidden Nodes: [9-9-9-9], Optimizer: Adam, Runs: 3, No Logic | Comparing Normalization', 'Epoch', 'BCE Loss', 0.5, True),
        'Validation Accuracy' : ('Validation: Accuracy (Smoothing 0.5) | Hidden Nodes: [9-9-9-9], Optimizer: Adam, Runs: 3, No Logic | Comparing Normalization', 'Epoch', 'Accuracy', 0.5, True)
    }

    tbp.plot_multi_run_experiments(folder_path, subfolder_names_alias_smoothing, plot_vars_alias_smoothing, plot_vars_identifiers, save_path, num_runs_dict)

    # TODO: 
    # - write the vales from tbp.read_single_scalar_tensorboard() to CSV and also be able to load them from there
    # - make it possible to also put multiple plots into one figure (e.g. training and validation next to each other for one experiment)
    # - smoothing with moving average: add window size
