import torch.nn as nn
from .basic_block import BasicBlock


class Encoder(nn.Module):
    """MultiBPE Decoder"""

    def __init__(
        self,
        fasta_size,
        input_size,
        hidden_size,
        fc_act_fn,
        num_basic_blocks,
        cnn_act_fn,
        kernel_size,
        stride,
        dropout,
        rnn_act_fn,
        num_directions,
    ):
        super(Encoder, self).__init__()
        infeatures = fasta_size + input_size
        outfeatures = hidden_size * num_directions

        self.input_fn = nn.Sequential(
            nn.Linear(infeatures, outfeatures), getattr(nn, fc_act_fn)()
        )

        assert num_basic_blocks > 0
        self.num_basic_blocks = num_basic_blocks
        self.basic_block = nn.ModuleList()
        for i in range(self.num_basic_blocks):
            d = dropout if i < self.num_basic_blocks - 1 else 0.0
            self.basic_block.append(
                BasicBlock(
                    outfeatures,
                    outfeatures,
                    cnn_act_fn=cnn_act_fn,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=d,
                    rnn_act_fn=rnn_act_fn,
                )
            )

    def forward(self, sequence, h0):
        hid = self.input_fn(sequence)
        h0 = h0.permute(1, 0, 2).contiguous()
        for ifunc in self.basic_block:
            hid = ifunc(hid, h0)
        return hid
