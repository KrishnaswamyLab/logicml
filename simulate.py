import argparse
import numpy as np
from code.pipeline.utils import *
from code.pipeline.handler import *
from code.pipeline.argparser_handler import *
from code.pipeline.logic_simulator import *


if __name__ == '__main__':

    # NOTE: with this script you can start the simulation of some Verilog logic modules that were derived from the pipeline and where the test data has been saved as npy-arrays. 
    # Or you can use it with manually specified npy-arrays as test data of your own (has to, however, fit the right dimensions, etc.).
    # This can especially useful for debugging, when you want to look into how intermediate signals behave and what their values are. 

    # TODO: the following code is again just an example of how to use it, you must adapt the code yourself.

    # ---------------- Argparser Handling ----------------
    used_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Combining Machine Learning With Logic Synthesis - Simulation')
    args_handler = ArgsHandler(parser, used_args)
    parser = args_handler.get_parser()
    args = args_handler.get_args()

    # ---------------- Handler  ----------------
    handler = Handler(name=args.experiment_name, verbosity=args.verbosity, basepath=args.basepath, overwrite=args.handler_overwrite)

    # ---------------- Simulator  ----------------

    ls = LogicSimulator(handler, args, None)
    ls.load_test_data_snapshot('/absolute/path/to/logic_test_data')
    ls.add_logic_experiment('/absolute/path/to/final/verilog/logic/logicnet_final.v', intermediate_outputs=['a_signal_of_interest'], only_intermediate_outputs=True)
    s = ls.simulate(debug=True, include_features_debug=False)[0]
    ls.save_simulation_results_array(s, 'sim_array')
    ls.calculate_confmat_from_simulation_results(s, 2)