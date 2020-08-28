# LogicML

This repro provides experimental code, which is part of a research project conducted at [Krishnaswamy Lab at Yale University](https://www.krishnaswamylab.org). It is also related to the paper **"Making Logic Learnable With Neural Networks"**, [which can be found on arxiv](https://arxiv.org/abs/2002.03847). Please also refer to this paper for explanations concerning the methods and especially carefully read through the details provided in the appendix! Note that, however, the data sets covered in the paper are not part of the code release. Thus, only dummy data sets are provided in this repository with reference to where they can be found. Additionally, note that the parameters that are set as default in this repro are not even optimized for the given dummy data sets, since we expect that this is anyway not of interest. You can try out different configurations yourself if you want and get a feeling for the framework before integrating your own data use-cases. Also: for big data sets with a lot of features (such as image data sets, e.g. MNIST) the framework can be slow and the methods in general maybe unsuitable. We recommend using it with small biomedical data sets. 

Further note that the code in this repro is not an official package. Instead it rather works as a standalone framework for experimenting in the intersection of machine learning and logic circuits. Therefore, it also still contains open ToDos and remains being under development. However, it can also serve as inspiration on how to emit Verilog code from [PyTorch](https://github.com/pytorch/pytorch) neural network models and on how to interface Python code with the C-based logic tool [ABC](https://github.com/berkeley-abc/abc). While each method and class is documented, it is advised to read through the following explanations that provide detailed insights on how the code is organized and structured. 

## Documentation

### List of Contents

Directly jump to the content you are looking for.<br><br>
1. [Installation and Setup](#installation)<br>
    1.1. [Top-Level Folder](#toplevelfolder)<br>
    1.2. [Pipeline Folder](#pipelinefolder)<br>
    1.3. [Evaluation Folder](#evaluationfolder)<br>
    1.4. [ABC Synthesis Folder](#abcsynthesisfolder)<br>
    1.5. [LogicNet Folder](#logicnetfolder)<br>
    1.6. [Results Folder](#resultsfolder)<br>
2. [Overview and File Organization](#overview)<br>
3. [Getting Started Guide](#gettingstarted)<br>
4. [Data Handling](#data)<br>
5. [Training Neural Networks](#training)<br>
6. [Logic Creation Process](#logiccreation)<br>
7. [Experiment Organization and Evaluation](#experiments)<br>
8. [Tips, Tricks and Advices](#tips)<br>
    8.1. [ArgParser Handling and Experiment Name](#argparser)<br>
    8.2. [Snapshotting](#snapshotting)<br>
9. [Contributions](#contributions)<br>

---
<a name="installation"> </a>
### Installation and Setup

**NOTE:** The following steps only have to be done once (especially the compilations of ABC and LogicNet - hence, step 3-7 - since the compiled execution files are part of the `.gitignore` file and will be left untouched once updating the repository).

**Step 1:** Clone this git repository:
`git clone https://github.com/KrishnaswamyLab/logicml`
`cd logicml`

**Step 2:** If you want to use your Python 3.7 interpreter directly, please check that you have all packages pip-installed that are listed in the `requirements.yml`-file. Otherwise, if you want to create an anaconda environment with the requirements of `logicml`, you can use the following commands: 
`conda env create -n logicml_env -f requirements.yml`
`conda activate logicml_env` for opening the environment and `conda deactivate` for closing it

**Step 3:** Clone the following git repository into a top-level folder that is not the logicml git repository - this will create a copy of the ABC logic synthesis tools:
`git clone https://github.com/berkeley-abc/abc`
`cd abc`

**Step 4:** Compile ABC into a static library as described in the repro's `README`: 
`make libabc.a` 

**Step 5:** Copy the `libabc.a`-file and `abc`-file that was created in this repro

**Step 6:** Navigate back to the `logicml` repro into `logicml/code/abc_synthesis/` and replace the potentially existing `libabc.a`-file and `abc`-file in this folder

**Step 7:** Now navigate to the `logicml/code/logicnet/`-folder and compile two versions of the LogicNet code with different verbosities into two executables with the following commands (this will overwrite potentially existing `lgn` and `lgn_verbose` files). 
`gcc -DLIN64 -o lgn lgn.c`
`gcc -DLIN64 -o lgn_verbose lgn_verbose.c`

**Step 8:** Now, your installation should work! Try it out by using: `python run.py --lgn`
If nothing crashes, everything should work. Now it's recommended to keep on reading the text below, since it explains the framework.

---
<a name="overview"> </a>
### Overview and File Organization

After the installation you are nearly ready to go. To get an overview of the framework architecture, let's first take a look at the different files, their classes and main purposes.

---
<a name="toplevelfolder"> </a>
#### Top-Level Folder

| File | Description |
| --- | --- |
| `__init__.py` | This file only contains imports. When adding new files or classes and having import problems, you should probably check out what is happening here.|
| `run.py` | This file is a script that serves as the one you will always be excuting from the console. It interfaces with the pipeline and runs any possible experiment configuration, since the pipeline is influenced by the ArgParser arguments that are provided in the console.|
| `simulate.py` | This script can be used for simulating a logic circuit once more after the logic was compiled from a pipeline run with the `run.py` script (within that, it is, however, usually anyway also simulated). This can especially be useful when wanting to debug a circuit and wanting to see intermediate wires signals.|
| `setup.py` | This file is justed needed for installation purposes.|

---
<a name="pipelinefolder"> </a>
#### Pipeline and DataLoaders Folder

| File | Description |
| --- | --- |
| `dataloader/data_readers.py` | Here, all data loaders should be defined with one class for each data set. They should all inherit from the class `AbstractDataReader`, as this is also where batches are tied and the data is normalized.|
| `pipeline/agparser_handler.py` | In this file, the class `ArgsHandler` can be found, which is where all the arguments and parameters are defined that can have influence on the pipeline execution and the corresponding experiments. This is also were a default handling for different data set configurations takes place, where valid parameter combinations are asserted and where a summary text of all chosen configurations per experiment is created. Whenever you want to introduce a new parameter `new_argument` in the pipeline, you should define it here (including assertions, a line for the summary text and default settings) and then you can use `self.args.new_argument` anywhere in the pipeline.|
| `pipeline/binarized_modules.py` | Since one idea of the NN-logic-pipeline that remains to be investigated is how [binarized neural networks](https://arxiv.org/abs/1602.02830) can be integrated into this workflow, the neural network in the pipeline is defined in such a way that it can either have real-valued weights (the standard case) or binary weights (needs to be explicitly called from the outside). Within the file `binarized_modules.py`, helper classes are defined, which is copied code from [here](https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py ). The integration of binarized neural networks can be seen as something that is still under development, since the logic translation processes currently assumes fully-connected neural networks with real-valued weights and ReLU activation functions [(as mentioned in our paper)](https://arxiv.org/abs/2002.03847). Thus, while the logic translation would probably currently not be correct, you could already train basic fully-connected binarized neural networks with the framework.|
| `pipeline/handler.py` | Here, the `Handler` class is defined, which takes care of the basic experiment organization. It writes log-files, CSVs, time-logs and resource management files, creates folders or does the snapshotting of your models.|
| `pipeline/logic_processing.py` | The `LogicProcessing` class is the heart of Verilog code generation. Here, the neural networks and LogicNets are dumped into Verilog modules, which are connected with each other and where the shell-scripts are created, which are used to interface with other logic tools. Note that the random forest logic is handled separately in the `pipeline/random_forest.py` script. Also note that the emitted Verilog code is in general "word-level" code and follows more specific rules like e.g. using just one module per file and one assignment per line, since this is the only way that [ABC](https://github.com/berkeley-abc/abc) can handle Verilog files.|
| `pipeline/logic_simulator.py` | Initially it was planned to use [Verilator](https://www.veripool.org/wiki/verilator), which, however, turned out to be too slow and ended up compiling the circuits for hours. The reason is that the AND-Inverter-Graphs (AIGs) can get very large, due to translating complex circuits into more basic gates. Hence, the `LogicSimulator` class defined here, circumvents the compilation process by just traversing the AIG structure and by evaluating each node's equation for a given input combination. This is done until the final nodes deliver the final results, but also intermediate nodes' signals can be used for debugging as proposed in the `simulate.py` script of the top-level folder.|
| `pipeline/nn.py` | Here, the class `LogicNN` is defined, which is a [PyTorch](https://github.com/pytorch/pytorch) neural network that is used to create the logic. It is defined in such a way that it is always fully-connected, but in future skip-connections should be supported, which is why some parts to do so are already integrated. Also, currently only ReLU activation functions on hidden layers are supported (due to the logic translation process) and softmax or sigmoid activations on the last layer. You will, however, already find reference to htanh activation functions, due to the future support of [binarized neural networks](https://arxiv.org/abs/1602.02830), as previously mentioned. Also dropout and batch normalization can be used during the training process, but they do not influence the logic translation process, since these are operations that are difficult to translate. Thus, when wanting to create correct logic, both should probably not be considered. What is special about the definition of the neural network in here is that the steps of a forward pass of the data depend on whether logic is created in that moment or not. If the forward pass is part of a logic creation process in that moment, the activations at each node are collected and processed, while no weight updates take place. Otherwise, it works as usual and just the gradients are collected for the backward pass afterwards.|
| `pipeline/random_forest.py` | This is a script, which provides methods that are called by the `Trainer` class. They do the training of random forests and the translation into logic.|
| `pipeline/result_reporter.py` | Since the pipeline currently only supports classification tasks (either binary or multi-class), the `ResultReporter` is used for creating confusion matrices, plotting them and calculating all performance measures from them, e.g. accuracy, recall, precision, etc. |
| `pipeline/trainer.py` | This file might be one of the most important ones and maybe is the first one you should look at when wanting to understand what is happening within the pipeline. The `Trainer` class is where all the top-level routines are defined, from the neural network training and its concluding random forest or LogicNet training to all logic translations. However, this is also where random forests are trained directly on the data or where all models are in general tested (the model itself and its corresponding logic). Here the neural networks can also be saved as ONNX-files to be visualized with [Netron](https://github.com/lutzroeder/netron) or where snapshots can be created and loaded, e.g. for experimenting only with different logic parameters based on always the same trained neural network. **Once more: the best is, you just read through the file from top to beginning and you will probably already understand a lot.**|
| `pipeline/utils.py` | Here, you will find a few helper methods that are used in other scripts and classes. Especially, this is also where the quantization scheme is defined. |

---
<a name="evaluationfolder"> </a>
#### Evaluation Folder

The scripts in this folder are in general quite experimental and sketched, however, maybe not too well documented. They should rather be seen as a add-ons to help you organize your experiments than as explicit features of the pipeline.

| File | Description |
| --- | --- |
| `evaluation/csv_latex.py` | This script is used for printing code of Latex tables to the console, which can be copied and added to a Latex document. The tables are created from the results that the pipeline wrote into CSV-files. You can either create tables where each experiment run gets one row (*in the script named Top1*) or you can create tables, in which multiple runs across one experiment are summarized (with mean and standard deviation) to one row (*in the script named TopN*). The script is not yet perfect and you will probably not get around some manual editing of the tables. Nonetheless, this script can save you some time, when setting the parameters correctly. |
| `evaluation/experiment_designer.py` | Similar to the `evaluation/csv_latex.py` script, you can use the `evaluation/experiment_designer.py` script to print permutations of different ArgParser settings to the console. You can then create a shell-script, where you first add a line `cd /absolute/path/of/git/top_level_folder` and then paste the copied lines from the console. When then executing the shell-script, e.g. in a `tmux` session, the experiments are run sequentially. |
| `evaluation/tensorboard_handling.py` | This script was in the end not used much. It is, however, a sketch of how to read the data from tensorboard events of different experiments. Going further from this, you can create nice plots with matplotlib showing you learning curves across experiments that you want to compare to each other and also apply some smoothing. |

---
<a name="abcsynthesisfolder"> </a>
#### ABC-Synthesis Folder
 You will probably only touch this folder once during the installation process. Otherwise, there is no need to do anything here. This is just where the compiled [ABC](https://github.com/berkeley-abc/abc) library is located, such that the pipeline can find it and use it.

---
<a name="logicnetfolder"> </a>
#### LogicNet Folder
 Similar to the ABC-Synthesis folder, the LogicNet folder just contains the LogicNet code and its compiled versions. You will not need to change anything here or look into this deeply.

---
<a name="resultsfolder"> </a>
#### Results Folder
 The results folder is not yet visible when you clone this repro. However, already for your first experiment it will be created by the `Handler` class. In this folder a subfolder for each experiment name will be created that stores the created logic and all other necessary information, e.g. for snapshotting. This also means that when you re-run an experiment under the same experiment name, the results will be overwritten, which is usually no problem when quickly trying out settings. However, when doing evaluation runs you need to be careful. The only time you will be warned and explicitly would need to set an overwrite option is when you enable the snapshotting mode and are about to overwrite already saved models under the same name. 
 
 Besides a subfolder for each experiment, the results folder also contains a subfolder for all models and a subfolder for all summaries, for the reason that you can quickly look up different settings that you have have set for the experiments. The summaries are additionally also stored in the corresponding experiments' subfolders. Also a subfolder for all confusion matrix plots will be created and a subfolder for all tensorboard events, so that you can navigate to this folder in the console and run the tensorboard there. If not declared otherwise with the appropriate argument from the `ArgsHandler` class, the CSV-files containing the results are stored in the results folder without any subfolder.

---
<a name="gettingstarted"> </a>
### Getting Started

Once you installed everything and gained afirst overview of the pipeline's functionalities, you can navigate to the top-level folder of the git and start the `run.py` script: 
`python /absolute/path/to/git_repro/run.py`

This will just train a neural network for a few epochs on a dummy data set with a few default settings and without creating logic. It will run under the default experiment name *logicml* (check the subfolder created within the results folder). When this works without problems, you can start experimenting with settings from the `ArgsHandler` by just adding the corresponding argument to the console command. For example you can train the neural network, then translate it to an arithmetic circuit without any additional training steps and simulate it - all under a new experiment name *translation* - by using the following command: 
`python /absolute/path/to/git_repro/run.py --nn_translation --experiment_name translation`

The best is to check the settings in the `ArgsHandler` class that you can make and just try them out a bit. 

---
<a name="data"> </a>
### Data Handling

As mentioned earlier, you should introduce a new class for each data set within the `dataloaders/data_readers.py` script, where each class should inherit from the `AbstractDataReader` class. The best is to start with a copy of one of the existing classes and rename it to make sure that you do not forget to declare the needed `self`-attributes. Then you also need to define a method `load_np_arrays` in which the data (independent of being the training or test data) is loaded into a numpy array `features` with the columns being the features and the rows being the observations. Respectively, the same should be done for the `labels` array, such that each label row matches the observation row of the `features` array. The `AbstractDataReader` then takes care of everything else, such as shuffling the data and tying batches, in such a way that the test data set is balanced and close to the splits that were set with the `ArgsHandler` arguments. If you don't want this style of data processing and want to use the exact same training and test data observations each time, you might need to adapt the code yourself. 

---
<a name="training"> </a>
### Training Neural Networks

As mentioned earlier, for the logic creation the framework currently only supports fully-connected neural networks with ReLU activations on hidden layers and sigmoid or softmax activations on the last layer. Further, it only supports classification tasks, which can be binary or multi-class classification. A support of architectures with skip-connections and binarized neural networks is planned for future releases, which is why some references can already be found. 

Nonetheless, you can already influence the number of hidden layers and nodes per layer with the `--nodes` parameter. The number of layers is given by the number of numbers that you provide behind this argument, while the number of nodes per layer are given by the number itself. For example `--nodes 8 5 3` will create a network with three hidden layers that contain 8, 5 and 3 nodes. For more options to take influence on the neural networks and its training hyper-paramters, please refer to one of the following sections - [here](#argparser). 

---
<a name="logiccreation"> </a>

###  Logic Creation Process

As described in the paper [**"Making Logic Learnable With Neural Networks"**](https://arxiv.org/abs/2002.03847), the process of logic creation depends on multiple settings. In general, it can be said that there are four different routines (with the details again depending on some parameters): 
1. *Arithmetic Circuit*: A neural network is directly translated into logic with multipliers, adders, comparators and multiplexers. 
2. *Direct Random Forest*: A random forest is trained directly on the data and then translated into equivalent logic of mainly comparators and multiplexers. 
3. *Random Forest*: First a neural network is trained, then random forests are trained on sets of the net's activations and lastly, the random forests are again translated into logic of mainly comparators and multiplexers. 
4. *LogicNet*: Analogously, LogicNets are trained on a neural network's activations and are translated into logic of mainly comparators and multiplexers. 

When talking about "logic translation" it is meant that the Python models are first translated into Verilog code, which is then loaded into [ABC](https://github.com/berkeley-abc/abc) to convert the underlying logic into And-Inverter-Graphs (AIGs). Within this procedure, there are special rules that apply to the Verilog code generation in order for ABC to be able to read it. ABC accepts a very limited subset of word-level Verilog for combinational circuits with any operators described in Verilog standard, including arithmetic operators and MUXes. For example, ABC can read Verilog with word-level signed and unsigned multipliers and adders represented using symbols `*` and `+`, and bit-blast them into AIGs using a variety of architectures, including Booth encoding.  

The restrictions on word-level Verilog are as follows:
- one operator per line
- the result of each operator should be stored as a separate internal variable
- the bit-width of internal variables participating in the same operator should be matched
- both bit sampling (such as `Var[5:2]`) and concatenation (such as `{a, b}`) are considered separate operators and cannot be combined with others on the same line (for example, ABC can read  `wire [15:0] a, b, c; assign c = a + b;` but it cannot read `wire [15:0] a, b, c; assign c = a + b[7:0];`)
- no user hierarchy, that is, only one top-level module and no instances of other modules

Word-level MUXes with bit-level controls can be presented using `?:` operator, for example:
```
wire [15:0] a, b, c;
wire ctrl;
assign c = ctrl ? a : b;
```

Word-level MUXes with word-level controls can be represented as shown below:
```
wire [1:0] ctrl;
wire [7:0] d0, d1, d2, d3, res;
  always @( ctrl or d0 or d1 or d2 or d3 )
    begin
    case ( ctrl )
      0 :  res = d0 ;
      1 :  res = d1 ;
      2 :  res = d2 ;
      3 :  res = d3 ;
    endcase
  end
```

For this reason, the Verilog code generation is done in a way that multiple word-level modules are created in different files. In the end they are connected by the code of each module being copied into one large file that contains just one final module. New wires are introduced to connect the copied modules' signals. Thus, the code creation can be seen as forming one large circuit from multiple building blocks.

---
<a name="experiments"> </a>

### Experiment Organization and Evaluation

As [mentioned earlier](#resultsfolder), you can give each experiment a name by using the argument `--experiment_name <my_experiment_name>` in the console behind the `python run.py` command. A new subfolder in the results folder will be created that contains all relevant data and information. You can also execute multiple runs of the same experiment setup by using the parameter `--num_runs <my_desired_number_of_runs>` in the console. This will change the experiment name by appending the current run's index to it for each run. However, the results of multiple runs and multiple different experiment setups can be written into the same CSV file. Therefore, you have the opportunity to structure your evaluations into blocks of experiments that you want to compare against each other. There are three different types of CSV-files that you can give different names to by using the appropriate commands: 
- `--nn_csv <name_of_csv_file>`: controls the name of the CSV file of regular neural network training results
- `--logic_csv <name_of_csv_file>`: controls the name of the CSV file of any created logic (arithmetic circuit, random forest, LogicNet or direct random forest)
- `--direct_rf_csv <name_of_csv_file>`: controls the name of the CSV file of regular direct random forest training

For all of those CSV files you can use the `--results_subfolder <name_of_subfolder>` command to bundle some CSV files in a subfolder of your choice within the results folder. A typical evaluation workflow for creating a number of experiments is the following:
1. Use the `evaluation/experiment_designer.py` script to get print-outs of parameter permutations in the console. Especially, use the CSV name parameters as just described. Note that you can set a prefix for the experiment name in this script and parameters you don't want to be a part of the experiment name. Then the experiment name is created automatically.
2. Copy the commands to a shell-script and start it. This will execute the experiments sequentially. Note that it makes sense to use copy it into multiple shell-scripts and start them in `tmux` sessions, since the experiments can take up some time.
3. Use `evaluation/experiment_designer.py` script to create Latex tables from the results in the CSV files.

---
<a name="tips"> </a>
### Tips, Tricks and Advices

The following points provide common issues, tips, tricks and advices.<br>

---
<a name="argparser"> </a>
##### ArgParser Handling:
There is a number of parameters available to make changes in the pipeline's execution. You can, however, always introduce your own new parameters by adding them to the `ArgsHandler` and creating assertions for the value ranges that you want to allow and for combinations with other parameters. Additionally you should add a description line to the `ArgsHandler.__repr__(self)` method, in order to show the setting in the summary file. Once you did all of this. You can use `self.args.my_new_argument` anywhere in the pipeline to reference it.

In the following, you will find tables that give an overview of the different parameters that the `ArgsHandler` already provides, in order to modify processes within the pipeline. In the console, use all mentioned parameters with `--` in advance, e.g. `--experiment_name` instead of `experiment_name`. When there is a doubled entry in the first column, this means that both names can be used in the console, since the ones after the first one are aliases of the first name.

In the following, you will find all **arguments that form general settings**.

| Parameter | Description | Values |
| --- | --- | --- |
| `experiment_name` | The name of this experiment - will create folder for results. | String |
| `no_cuda` | Disables CUDA training. | Boolean |
| `seed` | Random seed (default: 1) for CUDA training. | Integer |
| `log_interval` | How many batches to wait before logging training status. | Integer |
| `verbosity` | Sets the verbosity level for logging. | String - one of the following: [debug, info] |
| `basepath` | Basepath for handler to store results and logs (absolute path including last slash); if None: results folder in this GIT-Repro will be used. | String|
| `no_handler_overwrite` | Call when not wanting to overwrite the previous logs under same experiment name. | Boolean |
| `tensorboard_disable`<br>`tb_dis` | Parameter to set, if no tensorboard logging should be done. | Boolean |
| `num_runs` | Number of runs of an experiment. | Integer |
| `results_subfolder` | A string for a subfolder in the results folder, to which the results csv will be written. | String |
| `nn_results_filename`<br>`nn_csv` | A string for the csv file-name to which the results of neural network testing will be written. | String |
| `direct_rf_results_filename`<br>`direct_rf_csv`<br>`rf_csv` | A string for the csv file-name to which the results of direct random forest testing will be written. | String |
| `resource_monitoring`<br>`rm` | Parameter to set, if resource management should actively be monitored. | Boolean |
| `snapshotting` | Enables snapshotting. | Boolean |
| `snapshotting_overwrite` | Can be called if an already existing snapshot that runs under the same experiment name should be overwritten. | Boolean |
| `snapshotting_best` | Can be called if besides latest, also the best model in terms of performance on validation data set should be snapshotted. | Boolean |
| `snapshotting_interval` | The epoch interval for snapshotting the latest and best models during training. | Integer |
| `snapshotting_criterion` | Defines the export criterion for snapshotting best net. Use following structure <max/min>_<sth in return_dict from net validation>, currently: loss or accuracy. | String - one of the following: [max_loss, min_loss, max_accuracy, min_accuracy] |
| `load_snapshot` | In case you want to load a snapshot before training it further, hand over the path to the folder where the snapshotted files are located (with or without last slash). Be careful, it reconstructs the network and data reader setting and will ignore other settings made! None: in case you dont want to load a snapshot. | String |
| `just_test` | Parameter to call if the network should not be trained, but just tested (e.g. in combination with loading a pre-trained snapshot). | Boolean |
| `netron_export`<br>`netron` | Parameter to call if the neural network structure should be exported to be visualized with NETRON (for architecture learning in each round). | Boolean |

In the following, you will find all **arguments that are related to the data**.

| Parameter | Description | Values |
| --- | --- | --- |
| `datareader`<br>`reader` | Which data set to use. | String - one of the following: [coimbra, mnist] |
| `mnist_use_args_test_splits`<br>`mnist_split` | Parameter to call if the test_split parameter from the args should also be used for MNIST. Otherwise just regular test data set. | Boolean |
| `batch_size`<br>`bs` | Input batch size for training and testing. | Integer |
| `batch_size`<br>`bs` | Input batch size for training and testing. | Integer |
| `test_split`<br>`ts` | The percentage of data that should be used as test data. | Float |
| `validation_split`<br>`vs` | The percentage of training data that should be used as validation data. | Float |
| `enforce_new_data_creation` | Enforcing that data arrays are not loaded from numpy array files when stored, but being overwritten. | Boolean |
| `data_normalization_mode`<br>`norm` | Mode of Data Normalization - Choose between: [0: no normalization, 1: Z-Score Normalization, 2: Min-Max-Scaling, 3: Max-Abs-Scaling, 4: Robust-Scaling, 5: Power-Transforming, 6: Quantile-Transforming, 7: Independent Normalization]. | Integer - one of the following: [0, 1, 2, 3, 4, 5, 6, 7] |
| `no_balanced_test_data`<br>`balanced` | Parameter to call if (in case of using the arg splits) the test data should not be perfectly balanced among the classes. | Boolean |
| `unbalanced_class_train_keep`<br>`unbalanced_keep` | The percentage of the most unbalanced class that should be kept for training, in case of the datareader creating a perfectly balanced test set. | Float |

In the following, you will find all **arguments that are related to the neural network itself.**.

| Parameter | Description | Values |
| --- | --- | --- |
| `architecture` | Which network to use as initialization. Currently only supporting fully-connected architecture, i.e. the following keyword: fcnn. | String - one of the following: [fcbnn] |
| `hidden_layer_outnodes`<br>`nodes` | Number of nodes per hidden and input layer for fcbnn architecture (except for last layer) [length of list decides for number of layers]. | Row of Integers |
| `batchnorm` | Parameter to call if batch normalization should be done. | Boolean |
| `dropout` | Parameter to call if external dropout should be applied. | Boolean |
| `no_bias` | Disables biases in the fully connected layers. | Boolean |
| `binary_weights` | Enables usage of binary weights in the fully connected layers, i.e. uses BinarizeLinear layers instead of nn.Linear layers (but still independent of activation function). | Boolean |
| `bnn` | Turns it into a binarized neural network, i.e. using binarized weights and htanh activation function. | Boolean |
| `stochastic_binarization` | Enables usage of stochastic binarization in weight binarization forward pass, otherwise: deterministic binarization is used (=sign function). | Boolean |
| `hidden_activation` | Which activation function to choose for layers that are not the last one. Choose between following keywords: htanh, relu. | String - one of the following: [relu, htanh] |
| `sigmoid` | When this parameter is set and a binary classification is used, then the last layer has just one node, sigmoid is used and binary cross entropy loss. | Boolean |

In the following, you will find all **arguments that are related to the neural network training.**.

| Parameter | Description | Values |
| --- | --- | --- |
| `epochs`<br>`N` | Number of epochs to train. | Integer |
| `loss` | The loss used for the network. Choose between following keywords: bce, default. The default is currently set to be cross-entropy loss, but you can add your own loss definitions. | String - one of the following: [bce, default] |
| `optimizer`<br>`optim` | The loss used for the network. Choose between following keywords: adam, sgd. | String - one of the following: [adam, sgd] |
| `lr` | Learning rate (default: 0.001). | Float |
| `no_lr_scheduling` | Disables learning rate scheduling. | Boolean |
| `momentum` | SGD momentum (default: 0.5). | Float |
| `weight_decay`<br>`decay` | SGD weight decay. | Float |
| `no_validation` | Disables validation. | Boolean |

In the following, you will find all **arguments that are related to the logic generation in general.**.

| Parameter | Description | Values |
| --- | --- | --- |
| `just_logic` | Parameter to call if the network should not be trained, but just logic should be created (e.g. in combination with loading a pre-trained snapshot). | Boolean |
| `nn_translation`<br>`nnt`<br>`nn_trans` | Parameter to call if the routine for training a NN, translating it to logic and simulating it should be executed. | Boolean |
| `random_forest`<br>`rf` | Parameter to call if the routine for training a NN, training random forest on activations, translating it to logic and simulating it should be executed. | Boolean |
| `random_forest_direct`<br>`rf_direct`<br>`direct_rf` | Parameter to call if a random forest should be trained directly on the data and then be translated to logic (without involving neural networks). | Boolean |
| `logic_net`<br>`lgn` | Parameter to call if the routine for training a NN, training LogicNet on activations, translating it to logic and simulating it should be executed. | Boolean |
| `full_comparison`<br>`full`<br>`full_logic` | Parameter to call if all of the following args should be set to True at the same time: random_forest, logic_net, nn_translation. | Boolean |
| `aig_optimization`<br>`aig_optim` | Parameter to call if AIG statistics should be used after optimizing the AIG, otherwise statistics of original AIG will be used. | Boolean |
| `aig_optimization_command`<br>`aig_optim_cmd` | The command from ABC with which to opimtimize the AIG. Choose between following keywords: syn2, dc2, mfs, mfs_advanced | String - one of the following: [syn2, dc2, mfs, mfs_advanced] |
| `aig_export` | Parameter to call if final AIG file should be stored. | Boolean |
| `abc_verilog_export` | Parameter to call if ABC should export the processed Verilog file. | Boolean |
| `logic_results_filename`<br>`logic_csv` | A string for the csv file-name to which the results of logic testing will be written. | String |
| `total_bits`<br>`tb` | The number of total bits to which the activations, weights and biases should be quantized. | Integer |
| `fractional_bits`<br>`fb` | The number of bits from the total bits that should be used as fractional bits to which the activations, weights and biases should be quantized. Needs to be smaller than number of total bits. | Integer |
| `argmax_greater_equal`<br>`argmax` | Parameter to call if in final argmax of the logic, a comparison with greater equal should be done instead of just greater. | Boolean |
| `no_full_logic_report` | Parameter to call if no full readable logic report should be created. | Boolean |
| `blockwise_logic_report` | Parameter to call if blockwise, readable logic report should be created, i.e. per logic module. | Boolean |
| `no_logic_simulation`<br>`no_sim`<br>`no_simulation` | Parameter to call if the created logic should not be simulated, which might make sense if e.g. just the logic file is wanted or the report should be derived. | Boolean |
| `no_logic_test_data_snapshot` | Parameter to call if the test data should not be snapshotted as numpy arrays for potential future simulations. | Boolean |

In the following, you will find all **arguments that form LogicNet settings.**.

| Parameter | Description | Values |
| --- | --- | --- |
| `lgn_depth` | The number of layers in each LogicNet block. | Integer |
| `lgn_width` | The number of LUTs in each layer in each LogicNet block. | Integer |
| `lgn_lutsize` | The LUT-size in each LUT in each LogicNet block. | Integer |
| `lgn_lutsize_automatic`<br>`lutsize_auto` | Parameter to call if for LogicNet the LUT-Size of each module should automatically be defined by the number of inputs to the module, i.e. ignoring args.lgn_lutsize. | Boolean |
| `lgn_keep_intermediate_files` | Parameter to call if for LogicNet all intermediately created files should not be deleted (.data files, flist-file, shell-script, log-files, bit-wise Verilog modules). This can be helpful for debugging. | Boolean |

In the following, you will find all **arguments that form random forest settings.**.

| Parameter | Description | Values |
| --- | --- | --- |
| `rf_estimators` | The number of estimators (decision trees) in the random forest. | Integer |
| `rf_max_depth` | The maximal depth of the trees in the random forest. | Integer |
| `rf_keep_intermediate_files`<br>`rf_keep` | Parameter to call if for Random Forest all intermediately created files should not be deleted (bit-wise Verilog modules). This can be helpful for debugging. | Boolean |
| `rf_create_text_files`<br>`rf_text` | Parameter to call if for Random Forest the learned rules from the decision trees should be written to file. | Boolean |
| `rf_no_inverse_weighting`<br>`rf_no_weight` | Parameter to call if for Random Forest the under-represented classes should not be upweighted in decision trees. | Boolean |
| `rf_no_bitwise_training`<br>`rf_no_bitwise` | Parameter to call if for Random Forest the training and logic module derivation should not be done bitwise. | Boolean |
| `rf_threshold_upper`<br>`rf_upper`<br>`rf_thresh` | Parameter to call if for a non-bitwise Random Forest training the next higher integer of the float threshold should be chosen (otherwise lower). | Boolean |

---
<a name="snapshotting"> </a>
##### Snapshotting:
You can create snapshots of trained neural networks and later load them again. A snapshot contains all settings you made for the experiment, same as the exact splits of data points into training, validation and test set, which is why snapshots can get large in terms of memory. The snapshotting feature for example allows to first find the best neural network in terms of performance and then use it as input to find the best setting for creating logic. When enabling the `--snapshotting` option, a subfolder will be created under `results/models/your_current_experiment_name` that will again contain a subfolder `latest` and additionally `best` if you also set the `--snapshotting_best` option. In the `latest` folder, always the latest model will be overwritten, when a snapshot is created, which depends on the `--snapshotting_interval` option you chose. The "best" model will always be overwritten in the `best` folder, whenever a better model than the current one occurs. What "best" means in this context depends on what setting you chose for the `snapshotting_criterion` option - this can be one of the following: `min_loss`, `max_loss`, `min_accuracy` or `max_accuracy`. You can, however, add other options if you need them. Further, note that if you test your model in the end with the automatic pipeline execution, the *latest* model will be tested not the *best* one. To test the *best* one, you would need to load the corresponding snapshot and just apply another round of testing. 

Lastly, to load a snapshot, e.g. the *latest* one, not the *best* one, you would just use `python run.py --load results/models/your_current_experiment_name/latest`. Note that you probably need to choose another new experiment name that differs from the snapshot's name. If you would use the same experiment name, would load the snapshot and still enable the `--snapshotting` option, the execution would stop, in order to not overwrite the current snapshot, unless you explicitly allow it with `--snapshotting_overwrite`. When loading a snapshot, you can by the way in general still apply other new settings that do not apply to the neural network's structure itself.

---
<a name="contributions"> </a>
## Contributions 

The code was written by Tobias Brudermueller (RWTH Aachen University) within a project that was supervised by Smita Krishnaswamy (Yale University) and co-supervised by Johannes Stegmaier (RWTH Aachen University). The code covering LogicNet under `logicml/code/logicnet` was written by Alan Mishchenko (University of California, Berkeley), who also helped out on integrating the ABC Logic Synthesis Tools into the pipeline. Further, it was collaborated with Sat Chatterjee (Google AI), who was a big support and help, whenever needed. 
