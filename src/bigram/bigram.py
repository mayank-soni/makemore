import string

import matplotlib.pyplot as plt
import torch

from src.data.names import Names


class BigramModel:
    """Bigram character-level language model for names dataset"""

    chars = list(string.ascii_lowercase)
    ends = "[ENDS]"

    def __init__(self, names: Names, reg: int = 1):
        """
        Args:
            names (Names): Data class for names dataset
            reg (int, optional): Regularisation parameter. Defaults to 1.
        """
        self.names = names
        self.chars = self.get_chars()
        self.index2char, self.char2index = self.get_mapping()
        self.reg = reg
        self.bigram_array = torch.Tensor
        self.bigram_probs = torch.Tensor

    def get_chars(self) -> list[str]:
        """Starts with lowercase English letters, adds any other characters in names"""
        self.chars = BigramModel.chars
        for name in self.names.data.keys():
            for char in name:
                if char not in self.chars:
                    self.chars.append(char)
        # Add start/end of word character
        if BigramModel.ends not in self.chars:
            self.chars.append(BigramModel.ends)
        return self.chars

    def get_mapping(self) -> tuple[dict[int, str], dict[str, int]]:
        """Creates mappings between characters and indices for later use"""
        self.index2char = {i: char for i, char in enumerate(self.chars)}
        self.char2index = {char: i for i, char in enumerate(self.chars)}
        return self.index2char, self.char2index

    def train(self):
        """Trains bigram model on training data. See create_bigram_array for details.
        Also creates a matrix of probabilities from the bigram_array (normalised by row)
        Adds regularisation to avoid zero probabilities
        """
        self.bigram_array = self.create_bigram_array(self.names.train) + self.reg
        self.bigram_probs = self.bigram_array / self.bigram_array.sum(
            axis=1, keepdims=True
        )

    def create_bigram_array(self, names: dict[str, int]) -> torch.Tensor:
        """Creates bigram array from list of names.
        In effect, counts number of times each character appears after another and stores counts in a matrix.
        Rows are first character, columns are second character.
        Last row/column are for start/end of word.
        """
        # Initialise bigram array to zeros. Add one row/column to account for start/end of word
        bigram_array = torch.zeros(len(self.chars), len(self.chars))
        for (ch1, ch2), count in BigramModel.create_bigrams(names).items():
            bigram_array[self.char2index[ch1]][self.char2index[ch2]] += count
        return bigram_array

    @classmethod
    def create_bigrams(cls, names: dict[str, int]) -> dict[tuple[str, str], int]:
        """Creates list of bigrams from list of names.
        Start/end of word is represented by None.
        """
        bigrams = {}
        for name, count in names.items():
            name = [BigramModel.ends] + list(name) + [BigramModel.ends]
            for ch1, ch2 in zip(name, name[1:]):
                if (ch1, ch2) in bigrams:
                    bigrams[(ch1, ch2)] += count
                else:
                    bigrams[(ch1, ch2)] = count
        return bigrams

    # def visualise(self):
    #     """Visualises bigram array as a heatmap"""
    #     plt.figure(figsize=(16, 16))
    #     plt.imshow(self.bigram_array, cmap="Blues")
    #     for i in range(self.bigram_array.shape[0]):
    #         for j in range(self.bigram_array.shape[1]):
    #             first = self.index2char[i]
    #             # Shorten start/end of word to 'S'/'E' for readability
    #             if first == BigramModel.ends:
    #                 first = "S"
    #             second = self.index2char[j]
    #             if second == BigramModel.ends:
    #                 second = "E"
    #             chars = first + second
    #             # Need parsemath=False to avoid error with '$$'
    #             plt.text(
    #                 j, i, chars, ha="center", va="top", color="gray", parse_math=False
    #             )
    #             plt.text(
    #                 j,
    #                 i,
    #                 f"{self.bigram_array[i, j].item():.1g}",
    #                 ha="center",
    #                 va="bottom",
    #                 color="red",
    #             )
    #     plt.xlabel("Second Character")
    #     plt.ylabel("First Character")

    def predict_next(self, index: int, generator: torch._C.Generator) -> int:
        """Predicts next character given current character.
        Args:
            index (int): Index of current character
            generator (torch._C.Generator): Random number generator
        Returns:
            int: Index of next character
        """
        if generator:
            prediction = torch.multinomial(
                self.bigram_probs[index], 1, generator=generator
            ).item()
        else:
            # If no generator, use default random number generator
            prediction = torch.multinomial(self.bigram_probs[index], 1).item()
        return prediction

    def predict(
        self, generator: torch._C.Generator = None, max_length: int = None
    ) -> str:
        """Generates a name from the model.
        Args:
            generator (torch._C.Generator, optional): Random number generator. Defaults to None.
            max_length (int, optional): Maximum length of name to be generated. Defaults to None.
        """
        # Get index representing start/end of word
        ends_index = self.char2index[BigramModel.ends]
        word_indices = [ends_index]
        # Keep iteratively generating characters until end of word is predicted.
        # Or until max_length is reached
        while True:
            next_index = self.predict_next(word_indices[-1], generator)
            if next_index == ends_index:
                break
            word_indices.append(next_index)
            if max_length and len(word_indices) > max_length:
                break
        # Convert indices to characters, join up, and return
        return "".join([self.index2char[i] for i in word_indices[1:]])

    def evaluate(self) -> dict[str, float]:
        """Calculates loss on training and test data.
        Also calculates loss for baseline model with even probabilities."""
        # Get bigram array for test data
        test_array = self.create_bigram_array(self.names.test)
        # Get baseline probability array with even probabilities
        even_probs = torch.full((len(self.chars), len(self.chars)), 1 / len(self.chars))
        baseline_train_loss = self.loss(self.bigram_array, even_probs)
        baseline_test_loss = self.loss(test_array, even_probs)
        model_train_loss = self.loss(self.bigram_array, self.bigram_probs)
        model_test_loss = self.loss(test_array, self.bigram_probs)
        return {
            "baseline_train_loss": baseline_train_loss,
            "baseline_test_loss": baseline_test_loss,
            "model_train_loss": model_train_loss,
            "model_test_loss": model_test_loss,
        }

    def loss(self, bigram_array, probs) -> float:
        # average negative log-likelihood
        return -1 * torch.nansum(bigram_array * probs.log()) / (bigram_array.sum())
