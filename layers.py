"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        self.cnn = CNN(char_vectors.size(1), hidden_size)

    def forward(self, w, c):
        # w is (batch_size, seq_len)
        w_emb = self.embed(w)   # (batch_size, seq_len, embed_size)
        w_emb = F.dropout(w_emb, self.drop_prob, self.training)
        w_emb = self.proj(w_emb)  # (batch_size, seq_len, hidden_size)
        w_emb = self.hwy(w_emb)   # (batch_size, seq_len, hidden_size)

        """
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
            """
        # c is (batch_size, seq_len, max_word_len)
        c_emb = self.char_embed(c)  # (batch_size, seq_len, max_word_len, embed_size)
        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        c_emb = c_emb.permute(0, 3, 1, 2)  # (batch_size, embed_size, seq_len, max_word_len)
        c_emb = self.cnn(c_emb)  # (batch_size, seq_len, hidden_size)
        emb = torch.cat((w_emb, c_emb), 2)  # (batch_size, seq_len, 2 * hidden_size)
        return emb


class CNN(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv2d = nn.Conv2d(embed_size, hidden_size, (1, 5), bias=True)

    def forward(self, x):
        # x is size (batch_size, embed_size, seq_len, max_word_len)
        x = self.conv2d(x)  # (batch_size, hidden_size, seq_len, w_out)
        # padding=0, dilation=1, kernel_size[0]=1 so h_out=h_in=seq_len
        x = torch.max(F.relu(x), dim=-1)[0]  # (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)
        return x


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        # for each context word in each batch, have weighted sum of the question hidden states
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        # for each context word in each batch, have weighted sum of the context hidden states
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFSelfAttention(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFSelfAttention, self).__init__()
        self.drop_prob = drop_prob
        # self.w1 = nn.Parameter(torch.zeros(hidden_size, 1))
        # self.w2 = nn.Parameter(torch.zeros(hidden_size, 1))
        # self.v = nn.Parameter(torch.zeros(hidden_size, 1))
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.c_weight_2 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight_2 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight_2 = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.c_weight_3 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight_3 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight_3 = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.c_weight_4 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight_4 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight_4 = nn.Parameter(torch.zeros(1, 1, hidden_size))

        for weight in (self.c_weight, self.q_weight, self.cq_weight,
                       self.c_weight_2, self.q_weight_2, self.cq_weight_2,
                       self.c_weight_3, self.q_weight_3, self.cq_weight_3,
                       self.c_weight_4, self.q_weight_4, self.cq_weight_4):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, c_mask):
        # c is shape (batch_size, c_len, 2 * hidden_size)
        batch_size, c_len, _ = c.size()

        # 1. apply w1 and w2 as described in the paper (need to write it out in matrix form)
        # FOR NOW, actually just applying w_sim as in the original attention layer
        s = self.get_similarity_matrix(c, c, self.c_weight, self.q_weight, self.cq_weight)  # (batch_size, c_len, c_len)

        # 2. softmax
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        s_softmax = masked_softmax(s, c_mask, dim=2)  # check dim, we're trying to softmax the rows

        # 3. get attention output for every passage word
        # (bs, c_len, c_len) x (bs, c_len, 2 * hidden_size) => (bs, c_len, 2 * hidden_size)
        a = torch.bmm(s_softmax, c)

        # 2nd head
        s_2 = self.get_similarity_matrix(c, c, self.c_weight_2, self.q_weight_2, self.cq_weight_2)  # (batch_size, c_len, c_len)
        s_softmax_2 = masked_softmax(s_2, c_mask, dim=2)  # check dim, we're trying to softmax the rows
        a_2 = torch.bmm(s_softmax_2, c)

        # 3rd head
        s_3 = self.get_similarity_matrix(c, c, self.c_weight_3, self.q_weight_3, self.cq_weight_3)  # (batch_size, c_len, c_len)
        s_softmax_3 = masked_softmax(s_3, c_mask, dim=2)  # check dim, we're trying to softmax the rows
        a_3 = torch.bmm(s_softmax_3, c)

        # 4th head
        s_4 = self.get_similarity_matrix(c, c, self.c_weight_4, self.q_weight_4, self.cq_weight_4)  # (batch_size, c_len, c_len)
        s_softmax_4 = masked_softmax(s_4, c_mask, dim=2)  # check dim, we're trying to softmax the rows
        a_4 = torch.bmm(s_softmax_4, c)

        # done

        # 4. concatenate [c, x]
        x = torch.cat([c, a, a_2, a_3, a_4], dim=2)  # (bs, c_len, 4 * hidden_size)
        return x

    def get_similarity_matrix(self, c, q, c_weight, q_weight, cq_weight):
        """ Just performing w_sim^T[c_i; q_j; c_i * q_j] except c == q
        (Copied over from BidafAttention)
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, q_weight).transpose(1, 2)\
                                      .expand([-1, c_len, -1])
        s2 = torch.matmul(c * cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, num_heads, drop_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.multihead_attention = nn.MultiheadAttention(input_size, num_heads).to('cuda')

    def forward(self, x):
        # x: (batch_size, c_len, 2 * hidden_size)
        x = x.permute(1, 0, 2)  # (c_len, batch_size, 2 * hidden_size)
        attn_output, attn_output_weights = self.multihead_attention(x, x, x)
        return attn_output


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
