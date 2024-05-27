import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from typing import Tuple, List

class StreamingModelDataset(IterableDataset):
    def __init__(self, filename: str, steps: int = 1, sequence_length: int = 10) -> None:
        """
        Initialisation of the dataset. It loads a postprocessed_data.npz file in chunks.
        The targets of this file are divided by the amplification correction factor, so that
        data is made setup independent.

        Parameters
        ----------
        filename : str
            Folder and filename where the postprocessed_data.npz is.
        steps : int
            It allows to skip parts of the data when loading it into memory. The number indicates
            how many items will be skipped in between. The default is one (no values
            are skipped). E.g., if steps = 2, and the inputs are [0, 1, 2, 3, 4, 5, 6]. The only
            inputs taken into account would be: [0, 2, 4, 6].
        sequence_length : int
            The length of the RNN sequences.
        """
        self.filename = filename
        self.steps = steps
        self.sequence_length = sequence_length
        self.buffer_data = []
        self.load_data_config()

    def load_data_config(self):
        with np.load(self.filename, allow_pickle=True) as data:
            self.sampling_configs = dict(data["sampling_configs"].tolist())
            self.amplification = self.sampling_configs["driver"]["amplification"]

    def parse_data(self):
        with np.load(self.filename, allow_pickle=True) as data:
            inputs = data["inputs"][::self.steps]
            outputs = data["outputs"][::self.steps]
            combined_data = np.hstack((inputs, outputs))
            for start_idx in range(0, len(combined_data) - self.sequence_length):
                end_idx = start_idx + self.sequence_length
                if end_idx >= len(combined_data):
                    break
                input_seq = combined_data[start_idx:end_idx, :-1]
                target_value = combined_data[end_idx-1, -1] / self.amplification
                yield input_seq, target_value

    def __iter__(self):
        return self.parse_data()