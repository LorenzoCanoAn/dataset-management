from dataset_management.dataset_io import DatasetInputManager, DatasetOutputManager
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class DatasetFileManagerToPytorchDataset(Dataset):
    required_identifiers = []

    def __init__(
        self,
        name=None,
        mode="read",
        identifiers=dict(),
        unwanted_characteristics=dict(),
        **kwargs,
    ):
        self.name = name
        self.mode = mode
        if self.mode == "read":
            wanted_characteristics = {
                "dataset_type": self.__class__.__name__,
                "identifiers": identifiers,
            }
            if not name is None:
                wanted_characteristics["dataset_name"] = self.name
            self.input_manager = DatasetInputManager(
                wanted_characteristics, unwanted_characteristics
            )
        elif self.mode == "write":
            assert not name is None
            for required_identifier in self.required_identifiers:
                assert required_identifier in identifiers.keys()
            self.output_manager = DatasetOutputManager(
                self.name, self.__class__.__name__, identifiers
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
        return self._inputs[idx], self._labels[idx]

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
        n_samples = len(self.input_manager.file_paths)
        for i, file_path in tqdm(enumerate(self.input_manager.file_paths)):
            inpt, labl = self.read_sample(file_path)
            if i == 0:
                inpts_shape = [n_samples] + list(inpt.shape)
                labls_shape = [n_samples] + list(labl.shape)
                self._loaded_inputs = torch.zeros(inpts_shape)
                self._loaded_labels = torch.zeros(labls_shape)
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
