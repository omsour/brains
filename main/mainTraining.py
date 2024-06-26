import sys
import os
import yaml
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg') # To allow non-graphical plotting
from bspysmg.model.training import generate_surrogate_model
from brainspy.utils.io import load_configs
from bspysmg.data.postprocess import post_process
from bspysmg.model.lstm import LSTMModel

#inputs, outputs, info_dictionary = post_process('main\mainSamplingDataFull', clipping_value=None)
#print(f"max out {outputs.max()} max min {outputs.min()} shape {outputs.shape}")
      
"""
# Define the base directory for saving YAML files
results_base_dir = "main/mainTrainingData_HyperTuning"

# Define the ranges and steps for each parameter
sequence_length_range = range(10, 101, 20) 
hidden_size_range = range(16, 65, 16)  
num_layers_range = range(1, 3)  
learning_rate_range = [0.01, 0.008, 0.006]   

# Define the base YAML structure
base_yaml = {
    'results_base_dir': results_base_dir,
    'model_structure': {
        'type': 'LSTM',
        'input_features': 7,
        'sequence_length': None,
        'hidden_size': None,
        'num_layers': None,
        'D_in': 7,
        'D_out': 1
    },
    'hyperparameters': {
        'epochs': 80,
        'learning_rate': None
    },
    'data': {
        'dataset_paths': ["main/mainSamplingData/postprocessed_data.npz"],
        'steps': 1,
        'batch_size': 32,
        'worker_no': 0,
        'pin_memory': False,
        'split_percentages': [0.8, 0.1, 0.1]
    }
}

# Create the directory to save the YAML files if it doesn't exist
os.makedirs(results_base_dir, exist_ok=True)

for num_layers in num_layers_range:
    # Iterate over all combinations of the other parameters
    combinations = itertools.product(sequence_length_range, hidden_size_range, learning_rate_range)
    for idx, (sequence_length, hidden_size, learning_rate) in enumerate(combinations):
        # Update the base YAML structure with the current combination
        config = base_yaml.copy()
        config['model_structure']['sequence_length'] = sequence_length
        config['model_structure']['hidden_size'] = hidden_size
        config['model_structure']['num_layers'] = num_layers
        config['hyperparameters']['learning_rate'] = learning_rate
        
        # Define the filename based on the number of layers and combination index
        filename = f"config_layers_{num_layers}_{idx}.yaml"
        file_path = os.path.join('configs/training/Tuning', filename)
        
        # Save the YAML file
        with open(file_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file)

        print(f"Created: {file_path}")


for file_name in os.listdir('configs/training/Tuning'):
    if file_name.endswith('.yaml'):
        file_path = os.path.join('configs/training/Tuning', file_name)
        smg_configs = load_configs(file_path)
        generate_surrogate_model(smg_configs, custom_model=LSTMModel,main_folder= os.path.splitext(file_name)[0])

        print(f"Processed: {file_path}")
"""

smg_configs = load_configs('configs/fulltraining/smg_configs_template_omar.yaml')
generate_surrogate_model(smg_configs, custom_model=LSTMModel)