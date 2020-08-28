import numpy as np
import os
import sys
from threading import Timer
from psutil import cpu_percent, virtual_memory
from logging import (DEBUG, INFO, WARNING, FileHandler, Formatter, StreamHandler, getLogger, basicConfig)
import time
from filelock import Timeout, FileLock
from array import *
import json
from logicml.code.pipeline.utils import *

# ------------ Handler Class ------------

class Handler():
    def __init__(self, name='handler', verbosity='debug', basepath=None, overwrite=False, timeout=10):
        '''
            Intializes the Logger class.
            Args: 
                name: the name of the handler for logging, etc. - will also create corresponding folders 
                verbosity: the verbosity for logging ('debug' or 'info')
                basepath: (string) absolute path (including last slash) to where results should be stored, if None - will be created based on GIT-structure
                overwrite: Boolean to set to True if already existing files and paths should be deleted, when a handler with the same name and basepath already existed
                timeout: the time out for a lock for writing sth to file
        '''
        self.name = name 
        assert str.lower(verbosity) in ['debug', 'info'], 'Handler: Verbosity [{}] is not defined - only use one of the following [info, debug].'.format(verbosity)
        self.verbosity = str.lower(verbosity)

        self.timeout = timeout
        self.timers = {}
        self.timelog = {}
        self.resource_log = []
        self.resource_monitoring_active = False
        self.resource_monitoring_comment = None
        self.memory_warning_timestamp = 0


        # Basepath creation 
        if basepath is None: 
            self.basepath = self.create_basepath()
            self.tensorboard_path = self.create_tensorboardpath()
            self.results_path = self.create_results_path()
        else: 
            assert isinstance(basepath, str), 'Handler Class: Basepath can only of type string or None!'
            if not basepath.endswith('/'): 
                basepath += '/'
            
            self.basepath = basepath + self.name + '/'
            self.tensorboard_path = basepath + 'tensorboards/{}/'.format(self.name)
            self.results_path = basepath

        
        if overwrite: 
            self.delete_existing_files()

        mkdir(self.basepath)
        mkdir(self.tensorboard_path)
        mkdir(self.results_path)

        # Logger Creation 
        self.logger = None
        self.setup_logger()


    def delete_existing_files(self): 
        '''
            Deletes the files and subfolders in self.basepath if they do exist.
        '''
        
        os.system('rm -rf {}'.format(self.basepath))
        os.system('rm -rf {}'.format(self.tensorboard_path))

    def create_tensorboardpath(self): 
        '''
            Creates the path to store the tensorboard files (one folder up from basebath and in two subfolders)
        '''
        
        direct = os.path.dirname(self.basepath)
        direct = os.path.dirname(direct)
        direct += '/tensorboards/{}/'.format(self.name)
        return direct

    def create_results_path(self): 
        '''
            Create the basepath of the handler based on this GIT-structure and where the code is located.
        '''
        direct = os.path.dirname(os.path.abspath(__file__))
        direct = os.path.dirname(direct)
        direct = os.path.dirname(direct)
        direct += '/results/'
        return direct

    def create_basepath(self): 
        '''
            Create the basepath of the handler based on this GIT-structure and where the code is located.
        '''
        direct = os.path.dirname(os.path.abspath(__file__))
        direct = os.path.dirname(direct)
        direct = os.path.dirname(direct)
        direct += '/results/{}/'.format(self.name)
        return direct

    def setup_logger(self):
        '''
            Sets up the logger to later use it for any kind of logging. 
        '''
        if self.verbosity is None:
            self.logger = None
            return
        self.logger = getLogger(self.name)
        self.logger.setLevel(DEBUG)
        self.logger.propagate = False
        if not self.logger.handlers:
            fh = FileHandler(os.path.join(self.basepath, 'logging.log'))
            fh.setLevel(DEBUG)
            ch = StreamHandler(sys.stdout)
            ch_level = {'info': INFO, 'debug': DEBUG}[self.verbosity]
            ch.setLevel(ch_level)
            
            che = StreamHandler(sys.stderr)
            che.setLevel(WARNING)
            formatter = Formatter('[%(asctime)s {}] %(message)s'.format(self.name))
            formatter.datefmt = '%y/%m/%d %H:%M:%S'
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            che.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.logger.addHandler(che)

    def log(self, logtype, msg):
        '''
            Logs messages. 
            Args: 
                logtype: the type of logging message - can be one of the following: ['debug', 'info', 'warning', 'error']
                msg: The string to log
        '''
        if self.logger is None:
            return

        if logtype == 'debug':
            self.logger.debug('[DEBUG] {}'.format(msg))
        elif logtype == 'info':
            self.logger.info('[INFO] {}'.format(msg))
        elif logtype == 'warning':
            self.logger.warning('[WARNING] {}'.format(msg))
        elif logtype == 'error':
            self.logger.error('[ERROR] {}'.format(msg))

    def path(self, path=None, use_results_path=False, no_mkdir=False):
        '''
            Creates a path in sense of the Handler.
            This means that the Handler has a basepath where it is working on (e.g logging).
            The path will be 'concatenated' to the basepath and will create corresponding subfolders if they do not exist yet. 
            Args: 
                path: the path you want to adapt in the Logger's sense
                use_results_path: Boolean to set to True if a folder should not be created in the experiment folder, but above of that in the results folder
                no_mkdir: Boolean to set True if path should only be returned but folders should not be created in case of no existence
            Returns: 
                the 'concatenated' path in the handler folders

        '''
        if not use_results_path: 
            tp = self.basepath
        else: 
            tp = self.results_path
        if path is not None:
            tp = os.path.join(tp, path)
        dirpath = tp 
        if '.' in os.path.split(tp)[1]:
            dirpath = os.path.split(tp)[0]
        if not no_mkdir: 
            mkdir(dirpath)
        return tp + '/'

    def check_results_csv_file_path_exists(self, experiment_name, file_name, subfolder): 
        '''
            This method if the file that will be created within the write_results() method already exists. 
            This is needed for checking form the outside if this file already exists. 
            If not, then a first line can be written to the CSV that declares the column names. 
            NOTE: this method does nothing else than just returning the file_path - it doesn't create the file or write sth
            NOTE: therefore hand over the same parameters to this method as in write_results()

            Args: 
                experiment_name: subfolder within the 'subfolder' parameter of the current experiment folder
                file_name: file name (will be extended by .csv-ending)
                subfolder: subfolder within the current results folder, None for no subfolder
            Returns: 
                True if the file exists, False otherwise
        '''
        results_folder = os.path.dirname(self.basepath)
        results_folder = os.path.dirname(results_folder)

        if not subfolder: 
            filepath = '{}/{}.csv'.format(results_folder, file_name)
        else: 
            filepath = '{}/{}/{}.csv'.format(results_folder, subfolder, file_name)
        return os.path.isfile(filepath) 
    
    def write_result(self, columns, experiment_name, file_name='results', subfolder='csvfiles', writing_column_names=False):
        '''
            This method writes an experiment result . 

            Args: 
                columns: A list of column values that should be added
                experiment_name: subfolder within the 'subfolder' parameter of the current experiment folder
                file_name: file name (will be extended by .csv-ending)
                subfolder: subfolder within the current results folder, None for no subfolder
                writing_column_names: Boolean to set to True if column names are written (i.e. no addional timestamp and experiment name writing)
        '''

        results_folder = os.path.dirname(self.basepath)
        results_folder = os.path.dirname(results_folder)

        if not subfolder: 
            filepath = '{}/{}.csv'.format(results_folder, file_name)
        else: 
            filepath = '{}/{}/{}.csv'.format(results_folder, subfolder, file_name)
            mkdir(results_folder + '/{}/'.format(subfolder))

        lockpath = filepath + '.lock'
        lock = FileLock(lockpath, timeout=self.timeout)

        with open(filepath, 'a') as f:
            with lock: 
                if not writing_column_names: 
                    f.writelines('%f;%s;%s\n' % (time.time(), experiment_name, ';'.join([str(column) for column in columns])))
                else: 
                    f.writelines('%s\n' % (';'.join([str(column) for column in columns])))


    def tic(self, name='default'):
        '''
            This method is used for registering a process in the time management dictionary. 
            This means that it will add the start time of the process that you registered under the given name.

            Args: 
                name: name of the process
        '''
        self.timers.update({name: time.time()})

    def toc(self, name='default'): 
        '''
            This method is used for deregistering a process in the time management dictionary. 
            This means that it will add the difference of end time and start time of the process that you registered earlier under the given name.

            Args: 
                name: name of the process
        '''
        if name not in self.timelog:
            self.timelog.update({name: []})
        if name in self.timers:
            self.timelog[name].append(time.time() - self.timers[name])

    def write_timelog(self, name, add_to_previous_timelog=True, reset_timelog=True):
        '''
            This method writes a timelog in csv format of all processes that you registered with the tic() and toc() methods. 

            Args: 
                name: name of the timelog (will be added to the summary folder in the experiment folder)
                add_to_previous_timelog: boolean to set to True if an existing timelog should be extended
                reset_timelog: boolean to set to True if the handler's timelog should be reset after writing the csv
        '''
        filepath = self.path('summary/timelog.csv')
        filepath = filepath[:-1] # take away last slash
        tmp_dict = self.timelog.copy()
        res = ''

        if add_to_previous_timelog and os.path.exists(filepath):
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for n, l in enumerate(reader):
                    if len(l) <= 4:
                        continue
                    name = str(l[0])
                    mean = float(l[1])
                    total = int(l[2])
                    sums = float(l[3])
                    if name in tmp_dict:
                        mean = (mean * total + np.sum(tmp_dict[name])) / (total + len(tmp_dict[name]))
                        total += len(tmp_dict[name])
                        sums += np.sum(tmp_dict[name])
                        del tmp_dict[name]
                    res += '{};{};{};{}\n'.format(name, mean, total, sums)    
        for name in tmp_dict:
            res += '{};{};{};{}\n'.format(name, np.mean(tmp_dict[name]), len(tmp_dict[name]), np.sum(tmp_dict[name]))           

        with open(filepath, 'w') as csvfile:
            csvfile.write(res)

        if reset_timelog:
            self.timelog = {}

    def resource_monitoring(self, interval=1, live=False, warn=False):
        '''
            This method is used for keeping an eye on the CPU and memory resources and writing it to a csv file. 

            Args: 
                interval: resource checking interval in seconds (int)
                live: boolean - when set to True, the current status is directly written to the csv file under subfolder 'resources' at each interval (otherwise only when stoping monitoring)
                warn: boolean - when set to True, you will be warned when to much RAM is used
        '''
        if self.resource_monitoring_active:
            self.timer = Timer(interval, self.resource_monitoring, [interval, live, warn])
            self.timer.start()
        mem = virtual_memory()
        cpu = cpu_percent()
        cmt = '' if self.resource_monitoring_comment is None else self.resource_monitoring_comment
        measurement = [time.time(), cmt, cpu, mem[2]]
        if measurement[-1] > 99 and time.time() - self.memory_warning_timestamp > 60 * 60:
            self.memory_warning_timestamp = time.time()
        if live:
            filepath = self.path(os.path.join('summary/resources_live.csv'))
            filepath = filepath[:-1] # take away last slash
            line = ';'.join([str(v) for v in measurement]) + '\n'
            with open(filepath, 'a') as csvfile:
                csvfile.write(line)
        else:
            self.resource_log.append(measurement)
    
    
    def set_resource_comment(self, comment):
        '''
            This method is used for providing an additional comment to the current resource monitoring status.
            It will also be written into the csv for as long as you don't set a new resource comment.
        '''
        self.resource_monitoring_comment = comment

    def start_resource_monitoring(self, interval=1, live=False, warn=False):
        '''
            This method is used for starting the CPU and memory resource monitoring (setting it to active mode). 

            Args: 
                interval: resource checking interval in seconds (int)
                live: boolean - when set to True, the current status is directly written to the csv file under subfolder 'resources'
                warn: boolean - when set to True, you will be warned when to much RAM is used
        '''

        self.resource_monitoring_active = True
        self.resource_monitoring(interval=interval, live=live, warn=warn)

    def stop_resource_monitoring(self, name):
        '''
            This method is used for stoping the CPU and memory resource monitoring (setting it to inactive mode).
            It also writes everything collected so far to a csv file.
            Furthermore, the resource_monitoring_comment is reset. 

            Args: 
                name: the whole monitored resource process will be written to a csv file under this name under subfolder 'resources'
        '''

        self.resource_monitoring_active = False
        self.timer.cancel()
        self.resource_monitoring_comment = None
        if name is None:
            return
        filepath = self.path('summary/{}.csv'.format(name))
        filepath = filepath[:-1] # take away last slash
    
        lines = [';'.join([str(v) for v in vals]) for vals in self.resource_log]

        with open(filepath, 'w') as csvfile:
            csvfile.write('\n'.join(lines))

    def create_snapshot(self, snapshot_parameter_dict, subfolder='latest', filepath=None):
        '''
            This method is used for writing training snapshots in trainer.py to a json file. 

            Args: 
                snapshot_parameter_dict: a dictionary with parameters to write to the json file (must be handable by the json decoder)
                subfolder: subfolder within the experiment name folder (mostly: 'latest' or 'best')
                filepath: [optional] full path to the where the snapshot is located; if None, the standard path that depends on the experiment name is chosen
        '''
        if not filepath:
            filepath = self.path('models/{}/{}'.format(self.name, subfolder), use_results_path=True) + 'snapshot.json'
        else: 
            filepath = filepath

        self.log('debug', 'Acquiring snapshot')
        try:              
            with open(filepath, 'w') as f:
                json.dump(snapshot_parameter_dict, f)
        except json.decoder.JSONDecodeError as e:
            self.log('warning', 'Could not create snapshot: {}'.format(e))

        self.log('debug', 'Successfully acquired snapshot')


    def load_snapshot(self, subfolder='latest', filepath=None):
        '''
            This method is used for loading and restoring training snapshots in Trainer.py from a json file. 

            Args: 
                subfolder: subfolder within the experiment name folder (mostly: 'latest' or 'best')
                filepath: [optional] full path to the where the snapshot is located; if None, the standard path that depends on the experiment name is chosen
            Returns: 
                snapshot_parameter_dict: a dictionary with parameters in the format as used in the create_snapshot() method
        '''
        if not filepath:
            filepath = self.path('models/{}/{}'.format(self.name, subfolder), use_results_path=True) + 'snapshot.json'
        else: 
            filepath = filepath
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                parameter_dict = json.load(f)
            self.log('debug', 'Successfully loaded snapshot')
            return parameter_dict
        except Exception:
            self.log('debug', 'Could not load snapshot')
        return None
