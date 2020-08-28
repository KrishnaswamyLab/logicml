def print_experiment_string(experiment_names_prefix, experiment_names_excludings, fixed_args, varying_args_params, varying_args_bool):
    '''
        This method is a quick sketch for print-outs in the console that can be copied into an .sh-file.
        This can help when wanting to execute multiple sequential experiments with different setups. 
        Args: 
            experiment_names_prefix: a string with a prefix for all experiments
            experiment_names_excludings: a list of strings that should be excluded for the experiment name creation
            fixed_args: a string of fixed arguments for all experiments
            varying_args_params: a dictionary with string keys and lists of strings as values that together define the permutation of experiment settings
            varying_args_bool: a list of strings that are Boolean varying arguments for each experiment
    '''

    # --------- Assertions ----------------
    assert isinstance(experiment_names_prefix, str), 'Parameter experiment_names_prefix must be a string (which can also be empty if no prefix wanted).'
    assert isinstance(experiment_names_excludings, list), 'Parameter experiment_names_excludings must be a list.'
    for i in experiment_names_excludings: 
        assert isinstance(i, str), 'Elements in parameter experiment_names_excludings must be strings.'
    assert isinstance(fixed_args, str), 'Parameter experiment_names_prefix must be a string (which can also be empty if no fixed args wanted).'
    if not str.endswith(fixed_args, ' '): # last sign needs to be a space sign
        fixed_args = fixed_args + ' '
    assert isinstance(varying_args_params, dict), 'Parameter varying_args_params must be a dictionary with strings as keys and list of strings as values.'
    for k,v in varying_args_params.items(): 
        assert isinstance(k, str), 'Parameter varying_args_params must be a dictionary with strings as keys and list of strings as values.'
        assert str.startswith(k, '--'), 'The keys in the varying_args_params dictionary need to be strings that start with: --'
        assert isinstance(v, list), 'Parameter varying_args_params must be a dictionary with strings as keys and list of strings as values.'
        for i in v: 
            assert isinstance(i, str), 'Parameter varying_args_params must be a dictionary with strings as keys and list of strings as values.'
    assert isinstance(varying_args_bool, list), 'Parameter varying_args_bool must be a list of strings.'
    for i in varying_args_bool: 
        assert isinstance(i, str), 'Parameter varying_args_bool must be a list of strings.'
        assert str.startswith(i, '--'), 'The strings in the varying_args_bool list need to start with: --'

    # --------- The start ----------------

    beginning = 'python run.py'
    overall_string = ''
    string_collection = []
    flag = False

    # --------- Handle varying_args_params ----------------
    for k1, v1 in varying_args_params.items(): 

        # first elements creation
        if len(string_collection) == 0: 
            for v1_item in v1: 
                if 'layer_outnodes' in k1 or 'nodes' in k1: #needs to be handled differently because of argparser list format
                    s = beginning + ' {} {}'.format(k1, v1_item)
                else: 
                    s = beginning +  ' {}={}'.format(k1, v1_item)
                string_collection.append(s)

        # appending to the elements that are there already
        else:
            string_collection_new = []
            for s in string_collection:  
                copies = []

                for v1_item in v1:
                    if 'layer_outnodes' in k1 or 'nodes' in k1: #needs to be handled differently because of argparser list format
                        s_copy = s + ' {} {}'.format(k1, v1_item)
                    else: 
                        s_copy = s + ' {}={}'.format(k1, v1_item)
                    copies.append(s_copy)
                string_collection_new.extend(copies)
            
            string_collection = string_collection_new

        flag = False

    # --------- Handle varying_args_bool ----------------
    if len(varying_args_bool) != 0: 
        new_string_collection = []
        for v in varying_args_bool: 
            for s in string_collection: 
                new_string_collection.append(s)
                s_new = s + ' {}'.format(v)
                new_string_collection.append(s_new)

        string_collection = new_string_collection

    # --------- Adding the fixed arguments ----------------
    new_string_collection = []
    for s in string_collection: 
        new = s + ' {}'.format(fixed_args)
        new_string_collection.append(new)

    string_collection = new_string_collection

    # # --------- Handle the experiment name creation ----------------
    final_string_collection = []

    for s in string_collection: 

        final_experiment_name = experiment_names_prefix

        # a list of elements that still need to be parsed to form the final experiment name
        experiment_names_list = s.split('{} '.format(beginning))[-1].split('--')

        for i in range(1, len(experiment_names_list)): 
            experiment_name_element = experiment_names_list[i]

            # check whether this element should not be included in the experiment name 
            dont_use = False
            for exclude_name in experiment_names_excludings: 
                if exclude_name in experiment_name_element: 
                    dont_use = True 
                    break
            
            if not dont_use: 
                experiment_name_element = experiment_name_element.replace(' ', '_')
                experiment_name_element = experiment_name_element.replace('=', '')
                final_experiment_name = final_experiment_name + experiment_name_element
                final_experiment_name = final_experiment_name[:-1] + '*'

        final_string_collection.append(s + '--experiment_name={}'.format(final_experiment_name))

    # Removing the last sign because it will cause trouble in bash
    string_collection = []
    for s in final_string_collection: 
        string_collection.append(s[:-1])

    # --------- Final printable string creation ----------------
    for s in string_collection: 
        overall_string += s + '\n'

    print(overall_string)


if __name__ == '__main__':
    # This script produces print-outs of experiments and their argparser setups. 
    # Those can be copied to an .sh-file for sequential experiment evaluations. 
    # NOTE: The experiment names will be created automatically.
    
    # NOTE: The following code should just provide an example of how to use this script - it is not necessarily a configuration of experiments to consider.
    # The example in the following covers different settings of random forests logic creation.

    # NOTE: The code-words need to exist in form of argparser arguments and also the value ranges need to make sense.
    # This is only checked during runtime and in ase of misuse leads to the experiment not being executed. 
    # It is however not checked here when creating the print-outs

    # --------- Handle your setup here ----------------

    # parts that should be added to the experiment name in the beginning
    # NOTE: please include the * sign at the end
    experiment_names_prefix = '01_coimbra_rf_eval*'

    # parts that might be parameters of the experiment setup, but should actually not be part of the experiment name generation
    # needs to be a list of strings that can have more entries than the parameters that are actually used
    experiment_names_excludings = ['random_forest', 'verbosity', 'logic_csv', 'no_cuda', 'just_logic', 'no_vis', 'total_bits', 'fractional_bits', 'load_snapshot', 'tb_dis', 'num_runs', 'rf_text']

    # the fixed arguments for all of the experiments 
    # NOTE: include the last spacing 
    fixed_args = '--random_forest --logic_csv 01_coimbra_rf_eval --just_logic --no_cuda --tb_dis --verbosity=debug --total_bits 18 --fractional_bits 10 --num_runs=3 --rf_text '

 
    # the varying arguments that should be changed upon the permutation of experiments
    # NOTE: use '--' in the beginning of each key
    varying_args_params = {
        '--rf_estimators' : ['2', '3', '4'],
        '--rf_max_depth' : ['5', '10', '15', '20']
    }

    # the varying arguments that are booleans and should for each experiment setup be true and false each 
    varying_args_bool = ['--rf_no_bitwise'] 

    # the actual print-out
    print_experiment_string(experiment_names_prefix, experiment_names_excludings, fixed_args, varying_args_params, varying_args_bool)