import torch.nn as nn


class Decoder(nn.Module):
    """NetTIME decoder"""

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
        self.fc1 = nn.Linear(num_directions * hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc_act = getattr(nn, fc_act_fn)()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.fc_act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
