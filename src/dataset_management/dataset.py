from dataset_management.dataset_io import DatasetInputManager, DatasetOutputManager
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class DatasetFileManagerToPytorchDataset(Dataset):
    required_identifiers = []

    def __init__(
        self,
        datafolders_manager,
        name=None,
        mode="read",
        identifiers=dict(),
        unwanted_characteristics=dict(),
        samples_to_load=None,
        **kwargs,
    ):
        self.name = name
        self.mode = mode
        self.identifiers = identifiers
        self.samples_to_load = samples_to_load
        if self.mode == "read":
            wanted_characteristics = {
                "dataset_type": self.__class__.__name__,
                "identifiers": identifiers,
            }
            if not name is None:
                wanted_characteristics["dataset_name"] = self.name
            self.input_manager = DatasetInputManager(
                datafolders_manager, wanted_characteristics, unwanted_characteristics
            )
        elif self.mode == "write":
            assert not name is None
            for required_identifier in self.required_identifiers:
                assert required_identifier in identifiers.keys()
            self.output_manager = DatasetOutputManager(
                datafolders_manager, self.name, self.__class__.__name__, identifiers
            )
        self._loaded_labels = None  # Indexable
        self._loaded_inputs = None  # Indexable
        self._labels = None  # Indexable
        self._inputs = None  # Indexable
        self.import_args(**kwargs)
        if self.mode == "read":
            self.initialize_dataset()

    def new_env(self, path_to_env):
        if self.mode != "write":
            raise Exception("This method should only be called in write mode")
        self.output_manager.new_datafolder(path_to_env)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        inputs, labels = self._inputs[idx], self._labels[idx]
        return inputs, labels

    def write_datapoint(self, input_data, label):
        path = self.output_manager.get_path_to_new_train_sample()
        np.savez(path, input_data=input_data, label=label)

    def read_sample(self, path):
        sample = np.load(path)
        input_data = sample["input_data"]
        label = sample["label"]
        return input_data, label

    def load_dataset(self):
        print("Loading Dataset")
        if self.samples_to_load is None:
            paths_to_load = self.input_manager.file_paths
        else:
            paths_to_load = self.input_manager.file_paths[: self.samples_to_load]
        print(self.input_manager)
        n_samples = len(paths_to_load)
        self._loaded_inputs = [None for _ in range(n_samples)]
        self._loaded_labels = [None for _ in range(n_samples)]
        for i, file_path in tqdm(enumerate(paths_to_load),total=n_samples):
            inpt, labl = self.read_sample(file_path)
            self._loaded_inputs[i] = torch.Tensor(inpt)
            self._loaded_labels[i] = torch.Tensor(labl)

    def process_raw_inputs(self, *args, **kwargs):
        raise Exception("This function must be implemented in child class")

    def import_args(self, *args, **kwargs):
        raise Exception("This function must be implemented in child class")

    def initialize_dataset(self):
        print("Initializing dataset")
        self.load_dataset()
        self.process_raw_inputs()

    def get_loaded_inputs(self):
        return self._loaded_inputs

    def get_loaded_labels(self):
        return self._loaded_labels
