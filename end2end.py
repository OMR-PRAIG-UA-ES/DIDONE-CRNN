"""
Modules for the basic CRNN model used in the End-to-End modeling.
author: Adrian Roselló Pedraza (RosiYo)
"""

from types import SimpleNamespace
import numpy as np
import torch


class CNN(torch.nn.Module):
    """
    Standard Convolutional Neural Network for the End-to-End model.
    
    Args:
        num_channels (int): Number of input channels.
        img_height (int): Image height.
    """

    backbone: torch.nn.Sequential
    __num_channels: int
    __img_height: int
    __config: SimpleNamespace
    __height_reduction: int
    __out_channels: int

    def __init__(self, num_channels: int, img_height: int):
        super().__init__()
        self.__num_channels = num_channels
        self.__img_height = img_height

        self.__config = SimpleNamespace(
            filters=[num_channels, 64, 64, 128, 128],
            kernel=[5, 5, 3, 3],
            pool=[[2, 2], [2, 1], [2, 1], [2, 1]],
            leaky_relu=0.2,
        )

        layers = []
        for i in range(len(self.__config.filters) - 1):
            layers.append(
                torch.nn.Conv2d(
                    self.__config.filters[i],
                    self.__config.filters[i + 1],
                    self.__config.kernel[i],
                    padding="same",
                    bias=False,
                )
            )
            layers.append(torch.nn.BatchNorm2d(self.__config.filters[i + 1]))
            layers.append(torch.nn.LeakyReLU(self.__config.leaky_relu, inplace=True))
            layers.append(torch.nn.MaxPool2d(self.__config.pool[i]))

        self.backbone = torch.nn.Sequential(*layers)
        self.__height_reduction, self.__width_reduction = np.prod(
            self.__config.pool, axis=0
        ).tolist()
        self.__out_channels = self.__config.filters[-1]

    @property
    def num_channels(self) -> int:
        """[PROPERTY] Number of input channels."""
        return self.__num_channels

    @property
    def img_height(self) -> int:
        """[PROPERTY] Image height."""
        return self.__img_height

    @property
    def height_reduction(self) -> int:
        """[PROPERTY] Height reduction of the CNN."""
        return self.__height_reduction

    @property
    def width_reduction(self) -> int:
        """[PROPERTY] Width reduction of the CNN."""
        return self.__width_reduction

    @property
    def out_channels(self) -> int:
        """[PROPERTY] Number of output channels."""
        return self.__out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.backbone(x)


class RNN(torch.nn.Module):
    """
    Standard Recurrent Neural Network for the End-to-End model.
    
    Args:
        input_size (int): Input size of the RNN.
        output_size (int): Output size of the RNN.
    """

    __dropout: float
    blstm: torch.nn.LSTM
    linear: torch.nn.Linear

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.blstm = torch.nn.LSTM(
            input_size,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.__dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(256 * 2, output_size)

    @property
    def dropout(self) -> float:
        """[PROPERTY] Dropout rate."""
        return self.__dropout

    @dropout.setter
    def dropout(self, value: float):
        """[Setter] Dropout rate."""
        self.__dropout = torch.nn.Dropout(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x, _ = self.blstm(x)
        x = self.dropout(x)
        return self.linear(x)


class CRNN(torch.nn.Module):
    """CRNN model for the End-to-End model."""

    cnn: CNN
    __rnn_input_size: int
    rnn: RNN

    def __init__(self, num_channels, img_height, output_size):
        super().__init__()

        self.cnn = CNN(num_channels, img_height)
        self.__rnn_input_size = self.cnn.out_channels * (
            img_height // self.cnn.height_reduction
        )
        self.rnn = RNN(input_size=self.__rnn_input_size, output_size=output_size)

    @property
    def decoder_input_size(self) -> int:
        """[PROPERTY] Input size of the RNN."""
        return self.__rnn_input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.cnn(x)
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.__rnn_input_size)
        return self.rnn(x)
