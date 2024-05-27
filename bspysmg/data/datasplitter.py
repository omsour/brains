import numpy as np
import os
from typing import Tuple, List

class DataSplitter:
    def __init__(self, filename: str, split_percentages: List[float]):
        """
        Initializes the DataSplitter class.

        Parameters
        ----------
        filename : str
            Path to the postprocessed_data.npz file.
        split_percentages : list of float
            List containing the split percentages for training, validation, and test sets.
            E.g., [0.8, 0.1, 0.1] for 80% training, 10% validation, and 10% test sets.
        """
        assert sum(split_percentages) == 1.0, "Split percentages should add up to 1."
        self.filename = filename
        self.split_percentages = split_percentages

    def load_data(self) -> Tuple[np.array, np.array, dict]:
        """
        Loads the data from the npz file.

        Returns
        -------
        inputs : np.array
            Input data from the npz file.
        outputs : np.array
            Output data from the npz file.
        sampling_configs : dict
            Dictionary containing the sampling configurations.
        """
        with np.load(self.filename, allow_pickle=True) as data:
            sampling_configs = dict(data["sampling_configs"].tolist())
            inputs = data["inputs"]
            outputs = data["outputs"]
        return inputs, outputs, sampling_configs

    def split_data(self, inputs: np.array, outputs: np.array) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        """
        Splits the data into training, validation, and test sets.

        Parameters
        ----------
        inputs : np.array
            Input data to be split.
        outputs : np.array
            Output data to be split.

        Returns
        -------
        train_inputs : np.array
            Training input data.
        val_inputs : np.array
            Validation input data.
        test_inputs : np.array
            Test input data.
        train_outputs : np.array
            Training output data.
        val_outputs : np.array
            Validation output data.
        test_outputs : np.array
            Test output data.
        """
        total_len = len(inputs)
        train_end = int(self.split_percentages[0] * total_len)
        val_end = train_end + int(self.split_percentages[1] * total_len)

        train_inputs, val_inputs, test_inputs = np.split(inputs, [train_end, val_end])
        train_outputs, val_outputs, test_outputs = np.split(outputs, [train_end, val_end])

        return train_inputs, val_inputs, test_inputs, train_outputs, val_outputs, test_outputs

    def save_data(self, inputs: np.array, outputs: np.array, sampling_configs: dict, suffix: str):
        """
        Saves the split data into a new npz file.

        Parameters
        ----------
        inputs : np.array
            Input data to be saved.
        outputs : np.array
            Output data to be saved.
        sampling_configs : dict
            Dictionary containing the sampling configurations.
        suffix : str
            Suffix to add to the filename (e.g., 'training', 'validation', 'test').
        """
        new_filename = self.filename.replace('.npz', f'_{suffix}.npz')
        np.savez(new_filename, inputs=inputs, outputs=outputs, sampling_configs=sampling_configs)
        print(f"Saved {suffix} data to {new_filename}")

    def split_and_save(self):
        """
        Splits the data and saves the training, validation, and test sets into separate npz files.
        """
        inputs, outputs, sampling_configs = self.load_data()
        train_inputs, val_inputs, test_inputs, train_outputs, val_outputs, test_outputs = self.split_data(inputs, outputs)

        self.save_data(train_inputs, train_outputs, sampling_configs, 'training')
        self.save_data(val_inputs, val_outputs, sampling_configs, 'validation')
        self.save_data(test_inputs, test_outputs, sampling_configs, 'test')