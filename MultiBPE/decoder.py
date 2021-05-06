import torch.nn as nn


class Decoder(nn.Module):
    """MultiBPE decoder"""

    def __init__(
        self,
        hidden_size,
        output_size,
        seq_length,
        fc_act_fn,
        dropout,
        num_directions,
    ):
        super(Decoder, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(num_directions * hidden_size, hidden_size),
            getattr(nn, fc_act_fn)(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out = self.fn(x)
        return out
