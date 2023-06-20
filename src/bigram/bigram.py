from collections import Counter

import matplotlib.pyplot as plt
import torch

from src.config import ENDS


class BigramModel:
    """Bigram character-level language model for names dataset"""

    def __init__(self):
        self.index2char = dict[int, str]
        self.char2index = dict[str, int]
        self.reg = int
        self.bigrams = dict[tuple[str, str], int]
        self.bigram_array = torch.Tensor
        self.bigram_probs = torch.Tensor
        self.charlist = list[str]

    def train(
        self, bigrams: Counter[tuple[str, str]], charlist: list[str], reg: int = 1
    ) -> None:
        """Trains bigram model on set of bigrams. See create_bigram_array() for details.
        Also creates a matrix of probabilities from the bigram_array (normalised by row)
        Adds regularisation to reduce overfitting.
        """
        self.bigrams = bigrams
        self.charlist = charlist
        self.reg = reg
        (
            self.bigram_array,
            self.index2char,
            self.char2index,
        ) = BigramModel.create_bigram_array(self.bigrams, self.charlist, self.reg)
        self.bigram_probs = self.bigram_array / self.bigram_array.sum(
            axis=1, keepdims=True
        )

    @classmethod
    def create_bigram_array(
        cls, bigrams: Counter[tuple[str, str]], charlist: list[str], reg: int
    ) -> tuple[torch.Tensor, dict[int, str], dict[str, int]]:
        """Creates bigram array from bigram counts. In effect, converting Counter to matrix.
        Rows are first character, columns are second character.
        """
        # Initialise bigram array to zeros. Add one row/column to account for start/end of word
        index2char, char2index = BigramModel.get_mapping(charlist)
        bigram_array = (
            torch.zeros(len(charlist), len(charlist)) + reg
        )  # Regularise by adding reg to all counts
        for (ch1, ch2), count in bigrams.items():
            bigram_array[char2index[ch1]][char2index[ch2]] += count
        return bigram_array, index2char, char2index

    @classmethod
    def get_mapping(cls, chars) -> tuple[dict[int, str], dict[str, int]]:
        """Creates mappings between characters and indices for later use"""
        index2char = {i: char for i, char in enumerate(chars)}
        char2index = {char: i for i, char in enumerate(chars)}
        return index2char, char2index

    def predict(
        self, generator: torch._C.Generator = None, max_length: int = None
    ) -> str:
        """Generates a name from the model.
        Args:
            generator (torch._C.Generator, optional): Random number generator. Defaults to None.
            max_length (int, optional): Maximum length of name to be generated. Defaults to None.
        """
        # Get index representing start/end of word
        ends_index = self.char2index[ENDS]
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

    def evaluate(self, test_bigrams: Counter[tuple[str, str]]) -> dict[str, float]:
        """Calculates loss on training and test data.
        Also calculates loss for baseline model with even probabilities."""
        # Get bigram array for test data
        test_array = self.create_bigram_array(test_bigrams, self.charlist, 0)[0]
        # Get baseline probability array with even probabilities
        even_probs = torch.full(
            (len(self.charlist), len(self.charlist)), 1 / len(self.charlist)
        )
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

    # def visualise(self):
    #     """Visualises bigram array as a heatmap"""
    #     plt.figure(figsize=(16, 16))
    #     plt.imshow(self.bigram_array, cmap="Blues")
    #     for i in range(self.bigram_array.shape[0]):
    #         for j in range(self.bigram_array.shape[1]):
    #             first = self.index2char[i]
    #             # Shorten start/end of word to 'S'/'E' for readability
    #             if first == ENDS:
    #                 first = "S"
    #             second = self.index2char[j]
    #             if second == ENDS:
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
