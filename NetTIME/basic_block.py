import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block layer"""

    def __init__(
        self,
        infeatures,
        outfeatures,
        cnn_act_fn="ReLU",
        kernel_size=3,
        stride=1,
        dropout=0.0,
        rnn_act_fn="Tanh",
    ):
        super(BasicBlock, self).__init__()
        # CNN
        padding = int((kernel_size - 1) / 2)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                infeatures,
                outfeatures,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(outfeatures),
            getattr(nn, cnn_act_fn)(),
            nn.Conv1d(
                outfeatures,
                outfeatures,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(outfeatures),
        )

        self.cnn_act = getattr(nn, cnn_act_fn)()
        self.dropout_cnn = nn.Dropout(p=dropout)

        # RNN
        hidden_size = int(outfeatures / 2)
        self.rnn = nn.GRU(
            input_size=outfeatures,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.bn = nn.BatchNorm1d(outfeatures)
        self.rnn_act = getattr(nn, rnn_act_fn)()
        self.dropout_rnn = nn.Dropout(p=dropout)

    def forward(self, x, h0):
        # CNN with residual connection
        x = x.permute(0, 2, 1)
        residual = x
        out = self.cnn(x)
        out += residual
        out = self.cnn_act(out)
        out = self.dropout_cnn(out)
        out = out.permute(0, 2, 1)

        # RNN
        self.rnn.flatten_parameters()
        hid, _ = self.rnn(out, h0)
        out = self.bn(hid.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        out = self.rnn_act(out)
        out = self.dropout_rnn(out)

        return out
