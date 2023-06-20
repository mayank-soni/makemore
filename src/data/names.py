from collections import Counter
import pathlib
import random

import matplotlib.pyplot as plt
import torch

from src.config import DEFAULT_CHARS, ENDS


class Names:
    """Data class for names dataset."""

    def __init__(self):
        self.file_path = pathlib.Path
        self.data = Counter[str]
        self.train = Counter[str]
        self.test = Counter[str]
        self.chars = DEFAULT_CHARS
        self.all_bigrams = Counter[tuple[str, str]]
        self.train_bigrams = Counter[tuple[str, str]]
        self.test_bigrams = Counter[tuple[str, str]]
        self.x = torch.Tensor
        self.y = torch.Tensor
        self.x_train = torch.Tensor
        self.y_train = torch.Tensor
        self.x_test = torch.Tensor
        self.y_test = torch.Tensor
        self.weights = torch.Tensor
        self.weights_train = torch.Tensor
        self.weights_test = torch.Tensor

    @classmethod
    def from_file(cls, file_path: pathlib.Path, counts: bool = False) -> None:
        """Reads data from file. Allows for counts to be read in as well as names."""
        names = cls()
        names.file_path = file_path
        data = Counter()
        with open(file_path) as f:
            if counts:
                for line in f:
                    name, count = line.strip().lower().split(",")
                    data.update({name: int(count)})
            else:
                data.update({line.strip().lower(): 1 for line in f})
        names.data = data
        return names

    @classmethod
    def from_list(cls, namelist: list[str]) -> None:
        """Creates data from list of names"""
        names = cls()
        data = {name.strip().lower(): 1 for name in namelist}
        names.data = Counter(data)
        return names

    @classmethod
    def from_dict(cls, namedict: dict[str, int]) -> None:
        """Creates data from dict of names and counts"""
        names = cls()
        names.data = Counter(namedict)
        return names

    def train_test_split(self, train_size: float) -> None:
        """Splits data into train and test sets"""
        desired_train_count = round(self.data.total() * train_size, 0)
        current_train_count = 0
        # Shuffle data to randomise order for train-test split
        self.train = Counter()
        self.test = Counter()
        data_list = list(self.data.items())
        random.seed(36)
        random.shuffle(data_list)
        for i, (name, count) in enumerate(data_list):
            if current_train_count < desired_train_count:
                # Add name to train set if it will not exceed desired size
                if (current_train_count + count) < (
                    desired_train_count * 1.05
                ) or i == len(data_list) - 1:
                    # Add name to train set if it will not exceed desired size by more than 5%
                    # Except for last name in list, which is added to train set regardless
                    self.train.update({name: count})
                    current_train_count += count
                else:
                    self.test.update({name: count})
            else:
                self.test.update({name: count})

    @classmethod
    def to_bigrams(self, data: Counter[str] = None) -> Counter[tuple[str, str]]:
        """Creates datasets (train, test, total) of bigrams.
        Start/end of word is represented by ENDS.
        If data is passed, uses it instead and returns only one dataset.
        """
        bigrams = Counter()
        for name, count in data.items():
            name = [ENDS] + list(name) + [ENDS]
            bigrams.update({(ch1, ch2): count for ch1, ch2 in zip(name, name[1:])})
        return bigrams

    @classmethod
    def to_bigram_tensor(
        self, bigrams: Counter[tuple[str, str]], mapping: dict[str, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, weight = [], [], []
        for bigram, count in bigrams.items():
            x.append(mapping[bigram[0]])
            y.append(mapping[bigram[1]])
            weight.append(count)
        x = torch.tensor(x, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)
        weight = torch.tensor(weight, dtype=torch.int64)
        return x, y, weight

    def eda(self):
        """EDA on data"""
        lengths = Counter()
        sum_lengths = 0
        for name, count in self.data.items():
            lengths.update({len(name): count})
            sum_lengths += len(name) * count
        print("Max length: ", max(lengths.keys()))
        print("Min length: ", min(lengths.keys()))
        print("Avg length: ", sum_lengths / self.data.total())
        plt.hist(lengths.keys(), weights=lengths.values())
        plt.title("Length of Name")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.show()
        del lengths
