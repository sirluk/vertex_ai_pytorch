from torch import nn

from typing import Union


class ClfHead(nn.Module):

    ACTIVATIONS = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['gelu', nn.GELU()],
        ['tanh', nn.Tanh()]
    ])

    def __init__(self, hid_sizes: Union[int, list], num_labels: int, activation: str = 'tanh', dropout: bool = True, dropout_prob: float = 0.3):
        super().__init__()

        if isinstance(hid_sizes, int):
            hid_sizes = [hid_sizes]
            out_sizes = [num_labels]
        elif isinstance(hid_sizes, list):
            if len(hid_sizes)==1:
                out_sizes = [num_labels]
            else:
                out_sizes = hid_sizes[1:] + [num_labels]
        else:
            raise ValueError(f"hid_sizes has to be of type int or list but got {type(hid_sizes)}")

        layers = []
        for i, (hid_size, out_size) in enumerate(zip(hid_sizes, out_sizes)):
            if dropout:
                layers.append(nn.Dropout(dropout_prob))
            layers.extend([
                nn.Linear(hid_size, out_size),
                self.ACTIVATIONS[activation]
            ])
        layers = layers[:-1] # remove last activation

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()