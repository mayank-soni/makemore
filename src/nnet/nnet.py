import torch

from src.config import DEFAULT_CHARS


class CharNnet(torch.nn.Module):
    """Base class for character-level neural network.

    Attributes:
        layer_sizes (list[int]): List of layer sizes.
            First layer must be size len(DEFAULT_CHARS) for one-hot encoding.
        biases (bool): Whether to include biases in linear layers.
        reg (float): Regularisation strength for L2 regularisation.
        layers (list[dict[str, torch.Tensor]]): List of layers.
            Each layer is a dict containing the weight matrix
            and bias vector (if biases=True).
        loss_history (dict[str, list]): Dict containing lists of training and
            test losses.
    """

    def __init__(
        self, layer_sizes: list[int], biases: bool = True, reg: float = 0.0
    ) -> None:
        """Initialise weights and biases for each layer."""
        super().__init__()
        self.reg = reg
        generator = torch.Generator().manual_seed(36)
        self.layer_sizes = layer_sizes
        self.biases = biases
        self.layers: list[dict[str, torch.Tensor]] = []
        for i in range(len(layer_sizes) - 1):
            layer = {
                "W": torch.randn(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    generator=generator,
                    requires_grad=True,
                )
            }
            if self.biases:
                layer["b"] = torch.randn(
                    layer_sizes[i + 1], generator=generator, requires_grad=True
                )
            self.layers.append(layer)
        self.loss_history: dict[str, list] = {"train": [], "test": []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of neural network.

        Returns probabilities of each character in vocabulary

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, self.layer_sizes[0])

        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, self.layer_sizes[-1])
        """
        # Linear layer(s). Output shape for each: (batch_size, layer_size)
        for layer in self.layers:
            if self.biases:
                x = x @ layer["W"] + layer["b"]
            else:
                x = x @ layer["W"]
        return x

    def loss(
        self, probs: torch.Tensor, y: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss.

        Args:
            probs (torch.Tensor): Output probabilities from forward pass.
            y (torch.Tensor): Target values.
            weights (torch.Tensor): Weights for each sample in batch.

        Returns:
            torch.Tensor: Loss value.
        """
        # Extract probabilities for correct characters (i.e. each character in y)
        # for each row (each value in the arange), extract the yth value
        prob_correct = probs[torch.arange(len(y)), y]
        # Negative log likelihood loss
        loss = -torch.log(prob_correct)
        # Averaged over batch, weighted by counts
        loss = torch.sum(loss * weights) / weights.sum()
        # L2 regularisation. reg * average of squared weights from all layers
        if self.reg == 0.0:
            reg_loss = 0.0
        else:
            # L2 regularisation
            sum_param_squared = 0
            num_params = 0
            for layer in self.layers:
                sum_param_squared += self.reg * torch.sum(layer["W"] ** 2)
                num_params += torch.numel(layer["W"])
                if self.biases:
                    sum_param_squared += self.reg * torch.sum(layer["b"] ** 2)
                    num_params += torch.numel(layer["b"])
            # Average over number of parameters
            reg_loss = sum_param_squared / num_params
        return loss + reg_loss

    def update_weights(self, lr: float) -> None:
        """Update weights using gradients calculated during training.

        Args:
            lr (float): Learning rate.
        """
        with torch.no_grad():
            for layer in self.layers:
                layer["W"] -= lr * layer["W"].grad
                if self.biases:
                    layer["b"] -= lr * layer["b"].grad

    def zero_grad(self) -> None:
        """Reset gradients to zero."""
        for layer in self.layers:
            layer["W"].grad = None
            if self.biases:
                layer["b"].grad = None


class BigramNnet(CharNnet):
    """Bigram neural network. Inherits from CharNnet."""

    def __init__(self, layer_sizes: list[int], biases: bool, reg: float = 0) -> None:
        self._layer_sizes: list[int]
        self.layer_sizes = layer_sizes
        super().__init__(self.layer_sizes, biases, reg)

    @property
    def layer_sizes(self) -> list[int]:
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, layer_sizes: list[int]) -> None:
        """Check layer sizes before setting."""
        if layer_sizes[0] != len(DEFAULT_CHARS):
            raise ValueError(
                f"First layer must be size {len(DEFAULT_CHARS)} (size of output from one-hot encoding)"
            )
        if layer_sizes[-1] != len(DEFAULT_CHARS):
            raise ValueError(
                f"Last layer must be size {len(DEFAULT_CHARS)} for softmax"
            )
        self._layer_sizes = layer_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of neural network.
        Returns probabilities of each character in vocabulary (DEFAULT_CHARS))"""
        # Embedding via one-hot encoding. Input shape: (batch_size). Output shape: (batch_size, layer_sizes[0])
        x = torch.nn.functional.one_hot(x, num_classes=self.layer_sizes[0]).float()
        x = super().forward(x)
        return torch.softmax(x, dim=1)

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
        epochs: int,
        lr: float,
        val_x: torch.Tensor = None,
        val_y: torch.Tensor = None,
        val_weights: torch.Tensor = None,
        reporting_freq: int = 0,
    ):
        for i in range(epochs):
            # Reset gradients
            self.zero_grad()
            # Forward pass
            probs = self.forward(x)
            # Calculate loss
            train_loss = self.loss(probs, y, weights)
            self.loss_history["train"].append(train_loss.item())
            if val_x is not None and val_y is not None and val_weights is not None:
                val_loss = self.evaluate(val_x, val_y, val_weights)
                self.loss_history["test"].append(val_loss)
            # Backpropagation
            train_loss.backward()
            # Update weights
            self.update_weights(lr)
            if reporting_freq == 0:
                reporting_freq = epochs // 10
            if i % reporting_freq == 0:
                print(f"Epoch {i}: Train loss = {train_loss.item()}")
                # If validation data provided, print validation loss.
                # Else val_loss will be undefined, so skip.
                try:
                    print(f"Val loss = {val_loss}")
                except NameError:
                    pass

    def zero_grad(self) -> None:
        super().zero_grad()

    def update_weights(self, lr: float):
        super().update_weights(lr)

    def evaluate(self, x, y, weights):
        with torch.no_grad():
            probs = self.forward(x)
            loss = self.loss(probs, y, weights)
        return loss.item()

    def generate(
        self,
        mapping_i2c: dict[int, str],
        ends_index: int,
        max_length: int = None,
        generator: torch._C.Generator = None,
    ) -> str:
        word_indices = [ends_index]
        while True:
            next_index = self.predict_next(word_indices[-1], generator)
            if next_index == ends_index:
                break
            word_indices.append(next_index)
            if max_length and len(word_indices) > max_length:
                break
        # Convert indices to characters, join up, and return
        return "".join([mapping_i2c[i] for i in word_indices[1:]])

    def predict_next(self, index: int, generator: torch._C.Generator = None) -> int:
        with torch.no_grad():
            output_probs = self.forward(torch.tensor(index).unsqueeze(0))
        if generator:
            prediction = torch.multinomial(output_probs, 1, generator=generator).item()
        else:
            # If no generator, use default random number generator
            prediction = torch.multinomial(output_probs, 1).item()
        return prediction


class EmbeddingNet(CharNnet):
    def __init__(
        self,
        embedding_size: int,
        context_length: int,
        layer_sizes: list[int],
        biases: bool,
        reg: float = 0,
    ):
        self._layer_sizes: list[int]
        self.layer_sizes = layer_sizes
        super().__init__(self.layer_sizes, biases, reg)
        self.embedding_size = embedding_size
        self.embedding = torch.randn(
            len(DEFAULT_CHARS),
            embedding_size,
            generator=self.generator,
            requires_grad=True,
        )
        self.context_length = context_length

    @property
    def layer_sizes(self) -> list[int]:
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, layer_sizes: list[int]) -> None:
        if layer_sizes[0] != self.embedding_size * self.context_length:
            raise ValueError(
                f"First layer must be size {self.embedding_size * self.context_length} (embedding size x context length)"
            )
        if layer_sizes[-1] != len(DEFAULT_CHARS):
            raise ValueError(
                f"Last layer must be size {len(DEFAULT_CHARS)} for softmax"
            )
        self._layer_sizes = layer_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of neural network. Returns probabilities of each character in vocabulary"""
        x = self.embedding[x]
        x = super().forward(x.view(-1, self.embedding_size * self.context_length))
        return torch.softmax(x, dim=1)
