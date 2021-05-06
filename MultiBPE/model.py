import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class MultiBPE(nn.Module):
    """MultiBPE model"""

    def __init__(self, args):
        super(MultiBPE, self).__init__()
        ######## Configurations ########
        self.fasta_size = 4
        self.num_directions = 2

        self.tf_vocab_size = args.tf_vocab_size
        self.ct_vocab_size = args.ct_vocab_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.seq_length = args.seq_length
        self.embedding_size = args.embedding_size
        self.hidden_size = 2 * self.embedding_size

        self.fc_act_fn = args.fc_act_fn
        self.num_basic_blocks = args.num_basic_blocks
        self.cnn_act_fn = args.cnn_act_fn
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.dropout = args.dropout
        self.rnn_act_fn = args.rnn_act_fn

        ######## set up layers ########
        ## Embeddings ##
        self.ct_embed = nn.Embedding(args.ct_vocab_size, self.embedding_size)
        self.tf_embed = nn.Embedding(args.tf_vocab_size, self.embedding_size)
        self.fuse_ct = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
        )
        self.fuse_tf = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
        )

        ## Encoder ##
        self.encoder = Encoder(
            self.fasta_size,
            self.input_size,
            self.hidden_size,
            self.fc_act_fn,
            self.num_basic_blocks,
            self.cnn_act_fn,
            self.kernel_size,
            self.stride,
            self.dropout,
            self.rnn_act_fn,
            self.num_directions,
        )

        ## Dencoder ##
        self.decoder = Decoder(
            self.hidden_size,
            self.output_size,
            self.seq_length,
            self.fc_act_fn,
            self.dropout,
            self.num_directions,
        )

        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)

    def __get_embed_ids(self, data):
        return (data[:, 0], data[:, 1])

    def forward(self, data):
        x, ids = data
        tf, ct = self.__get_embed_ids(ids)

        tf_emb = self.tf_embed(tf).unsqueeze(1)
        ct_emb = self.ct_embed(ct).unsqueeze(1)

        h0 = torch.cat((tf_emb, ct_emb), dim=2)
        h0 = h0.repeat(1, self.num_directions, 1)

        hid = self.encoder(x, h0)
        y = self.decoder(hid)

        return y
