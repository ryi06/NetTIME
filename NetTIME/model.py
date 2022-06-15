import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class NetTIME(nn.Module):
    """NetTIME model"""

    def __init__(self, args):
        super(NetTIME, self).__init__()
        ######## Configurations ########
        self.fasta_size = 4
        self.num_directions = 2

        self.tf_vocab_size = args.tf_vocab_size
        self.ct_vocab_size = args.ct_vocab_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.seq_length = args.seq_length
        self.embedding_size = args.embedding_size

        self.fc_act_fn = args.fc_act_fn
        self.num_basic_blocks = args.num_basic_blocks
        self.cnn_act_fn = args.cnn_act_fn
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.dropout = args.dropout
        self.rnn_act_fn = args.rnn_act_fn

        # Disable TF or cell type embeddings.
        self.disable_tf_embed = args.disable_tf_embed
        self.disable_ct_embed = args.disable_ct_embed
        # TF and CT embedding cannot be disabled at the same time.
        if self.disable_tf_embed and self.disable_ct_embed:
            raise ValueError(
                "--disable_tf_embed and --disabled_ct_embed cannot both be"
                "True."
            )
        elif self.disable_tf_embed or self.disable_ct_embed:
            self.hidden_size = self.embedding_size
        else:
            self.hidden_size = 2 * self.embedding_size

        ######## set up layers ########
        ## Embeddings ##
        if not self.disable_tf_embed:
            self.tf_embed = nn.Embedding(
                args.tf_vocab_size, self.embedding_size
            )
        if not self.disable_ct_embed:
            self.ct_embed = nn.Embedding(
                args.ct_vocab_size, self.embedding_size
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

    def __init_random_embed(self, data):
        batch_size = data.shape[0]
        device = data.device
        emb = torch.randn(batch_size, self.embedding_size)
        return emb.to(device).unsqueeze(1)

    def forward(self, data):
        x, ids = data
        tf, ct = self.__get_embed_ids(ids)

        def get_embedding_vector(embed_id, embed_mat):
            return embed_mat(embed_id).unsqueeze(1)

        if self.disable_ct_embed:
            # Only TF embedding.
            h0 = get_embedding_vector(tf, self.tf_embed)
        elif self.disable_tf_embed:
            # Only CT embedding
            h0 = get_embedding_vector(ct, self.ct_embed)
        else:
            # TF and CT embedding
            tf_emb = get_embedding_vector(tf, self.tf_embed)
            ct_emb = get_embedding_vector(ct, self.ct_embed)
            h0 = torch.cat((tf_emb, ct_emb), dim=2)

        h0 = h0.repeat(1, self.num_directions, 1)

        hid = self.encoder(x, h0)
        y = self.decoder(hid)

        return y
