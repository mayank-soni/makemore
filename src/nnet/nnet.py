import torch

from src.config import DEFAULT_CHARS


class BigramNnet(torch.nn.Module):
    def __init__(self, layer_sizes: list[int], reg: float = 0):
        super().__init__()
        self.x = torch.Tensor
        self.y = torch.Tensor
        self.x_train = torch.Tensor
        self.y_train = torch.Tensor
        self.x_test = torch.Tensor
        self.y_test = torch.Tensor
        self.train_weights = torch.Tensor
        self.test_weights = torch.Tensor
        self.weights = torch.Tensor
        # L2 regularisation term
        self.reg = reg
        generator = torch.Generator().manual_seed(36)
        self._layer_sizes = list[int]
        self.layer_sizes = layer_sizes
        self.layers = {
            f"W{i}": torch.randn(
                layer_sizes[i],
                layer_sizes[i + 1],
                generator=generator,
                requires_grad=True,
            )
            for i in range(len(layer_sizes) - 1)
        }
        self.loss_history = {"train": [], "test": []}

    @property
    def layer_sizes(self) -> list[int]:
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, layer_sizes: list[int]) -> None:
        if layer_sizes[0] != len(DEFAULT_CHARS):
            raise ValueError(
                f"First layer must be size {len(DEFAULT_CHARS)} for one-hot encoding"
            )
        if layer_sizes[-1] != len(DEFAULT_CHARS):
            raise ValueError(
                f"Last layer must be size {len(DEFAULT_CHARS)} for softmax"
            )
        self._layer_sizes = layer_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of neural network. Returns probabilities of each character in vocabulary"""
        # Embedding via one-hot encoding. Input shape: (batch_size). Output shape: (batch_size, layer_sizes[0])
        x = torch.nn.functional.one_hot(x, num_classes=self.layer_sizes[0]).float()
        # Linear layer(s). Output shape for each: (batch_size, layer_size)
        for layer in self.layers.values():
            x = x @ layer
        return torch.softmax(x, dim=1)

    def loss(
        self, probs: torch.Tensor, y: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        # Extract probabilities for correct characters (i.e. each character in y)
        # for each row (each value in the arange), extract the yth value
        prob_correct = probs[torch.arange(len(y)), y]
        # Negative log likelihood loss
        loss = -torch.log(prob_correct)
        # Averaged over batch, weighted by counts
        loss = torch.sum(loss * weights) / weights.sum()
        # L2 regularisation. reg * average of squared weights from all layers
        if self.reg == 0:
            reg_loss = 0
        else:
            # L2 regularisation i.e.
            reg_loss = (
                self.reg
                * sum(torch.sum(layer**2).item() for layer in self.layers.values())
                / sum(torch.numel(layer) for layer in self.layers.values())
            )
        return loss + reg_loss

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
        reporting_freq: int = None,
    ):
        for i in range(epochs):
            # Reset gradients
            for layer in self.layers.values():
                layer.grad = None
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
            with torch.no_grad():
                for layer in self.layers.values():
                    layer -= lr * layer.grad
            if reporting_freq == None:
                reporting_freq = epochs // 10
            if i % reporting_freq == 0:
                print(f"Epoch {i}: Train loss = {train_loss.item()}")
                try:
                    print(f"Val loss = {val_loss}")
                except NameError:
                    pass

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
