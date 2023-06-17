import torch

from src.bigram.bigram import BigramModel
from src.data.names import Names


class BigramNnet(torch.nn.Module):
    def __init__(self, names: Names, reg: float = 0.1):
        super().__init__()
        self.names = names
        self.bigram = BigramModel(self.names)
        self.x_train, self.y_train, self.x_test, self.y_test = (
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        )
        # n_neurons needs to be the same as the number of potential output characters
        self.n_neurons = len(self.bigram.chars)
        # One-hot encoding of characters (embedding size = number of characters)
        self.n_embedding = len(self.bigram.chars)
        # Single hidden layer which is also the output layer. No biases, just weights
        self.layer = torch.randn(
            (self.n_embedding, self.n_neurons),
            generator=torch.Generator().manual_seed(36),
            requires_grad=True,
        )
        self.reg = reg

    def get_data(self):
        self.x_train, self.y_train = self.create_bigram_tensors(self.names.train)
        self.x_test, self.y_test = self.create_bigram_tensors(self.names.test)

    def create_bigram_tensors(self, names: list[str]) -> tuple[torch.Tensor]:
        x, y = [], []
        bigrams = BigramModel.create_bigrams(names)
        for bigram in bigrams:
            x.append(self.bigram.char2index[bigram[0]])
            y.append(self.bigram.char2index[bigram[1]])
        x = torch.tensor(x, dtype=torch.int64)
        x_enc = torch.nn.functional.one_hot(
            x, num_classes=len(self.bigram.chars)
        ).float()
        y = torch.tensor(y, dtype=torch.int64)
        return x_enc, y

    def forward(self, x):
        """Forward pass of a single layer."""
        # Neuron activations
        # n_samples x n_neurons = (n_samples x n_embedding) @ (n_embedding x n_neurons)
        output = x @ self.layer
        # Softmax for probs
        probs = torch.softmax(output, dim=1)
        return probs

    def loss(self, probs, y_true):
        # CCE Loss
        # Extract probabilities for correct characters (i.e. each character in y)
        # for each row (each value in the arange), extract the yth value
        prob_correct = probs[torch.arange(len(y_true)), y_true]
        # log the probs and take the negative and mean. Add L2 regularisation
        loss = (-1 * torch.log(prob_correct).sum() / len(y_true)) + (
            self.reg * torch.sum(self.layer**2)
        )
        return loss
