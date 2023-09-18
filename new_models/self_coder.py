import torch
import numpy as np

class PostureEncoder(torch.nn.Module):
    def __init__(self,
                 batch_size=32,
                 in_size=1,
                 hidden_size=34,
                 hidden_layers=16):
        super(PostureEncoder, self).__init__()

        self.batch_size  = batch_size
        self.input_size  = in_size
        self.hidden_size = hidden_size
        self.ReLU        = torch.nn.ReLU(inplace=True)
        self.Flatten     = torch.nn.Flatten()

        # input batch x pose x chan1 x chan2 68
        self.model_encoder_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, 32, 5, 1, 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
        )

        self.model_encoder_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 5, 1, 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, 2, 1),
            torch.nn.Dropout(0.5),
        )

        self.model_linear_block1 = torch.nn.Sequential(
            torch.nn.Linear(128 * (hidden_size // 4) * (hidden_size // 4), 32 * (hidden_size // 4) * (hidden_size // 4)),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
        )

        self.model_linear_block2 = torch.nn.Sequential(
            torch.nn.Linear(32 * (hidden_size // 4) * (hidden_size // 4), hidden_layers * in_size * (hidden_size // 4) * (hidden_size // 4)),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
        )


    def forward(self, x):
        z  = self.model_encoder_block1(x)
        z  = self.model_encoder_block2(z)

        z  = self.Flatten(z)

        z  = self.model_linear_block1(z)
        z  = self.model_linear_block2(z)

        return z

class PostureDecoder(torch.nn.Module):
    def __init__(self,
                 batch_size=32,
                 hidden_layers=16,
                 hidden_size=34,
                 out_size=1):
        super(PostureDecoder, self).__init__()

        self.batch_size    = batch_size
        self.hidden_layers = hidden_layers
        self.out_size      = out_size
        self.hidden_size   = hidden_size
        self.ReLU          = torch.nn.ReLU()
        self.Sigmoid       = torch.nn.Sigmoid()

        self.model_decoder_linear1 = torch.nn.Sequential(
            torch.nn.Linear(self.out_size * (self.hidden_size // 4) * (self.hidden_size // 4) * self.hidden_layers,
                             (self.hidden_size // 8) * (self.hidden_size // 8) * self.out_size * self.hidden_layers),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.8)
        )

        self.model_decoder_block1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_layers, out_size, 4, 2),
        )

        self.model_decoder_block2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(out_size, out_size, 4, 2, 3),
        )

        self.model_decoder_block3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(out_size, out_size, 4, 2),
        )

    def forward(self, x):
        x = self.model_decoder_linear1(x)
        x = torch.reshape(x, [-1, self.hidden_layers, (self.hidden_size // 8), (self.hidden_size // 8)])

        z = self.model_decoder_block1(x)
        z = self.model_decoder_block2(z)
        z = self.model_decoder_block3(z)

        return self.ReLU(z)
