import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

class MultiHeadAttention(nn.Module):
    #in_features: Size of each input sample.
    #head_num: Number of heads.
    #bias: Whether to use the bias term.
    #activation: The activation after each linear transformation.
    def __init__(self,
                 in_features,
                 head_num,
                 bias = True,
                 activation = None):
                 ##activation = F.relu):
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(EncoderBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size * heads,
                 head_num = heads,
                 bias = True,
                 activation = None)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        
        out = self.norm1(x)

        attention = self.MultiHeadAttention(out, out, out, mask)

        attention = attention + x

        out2 = self.norm2(attention)
        out3 = self.mlp(out2)

        out3 = out3 + attention

        return out3

class MultiLayerEncoder(nn.Module):
    def __init__(
        self,
        embed_size,
        reduced_dims,
        heads = 4,
        num_layers = 4,
        device = 'cpu'):

        self.use_bias = False

        super(MultiLayerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.layers = nn.ModuleList()

        # first Linear
        self.layers.append(nn.Linear(in_features = embed_size, out_features = reduced_dims[0], bias = self.use_bias))

        last_dim = embed_size
        # Encoder - Linear stack
        for dim in reduced_dims:

            # encoders
            for _ in range(num_layers):
                self.layers.append(EncoderBlock(embed_size = last_dim, heads = heads))

            # linear
            self.layers.append(nn.Linear(in_features = last_dim, out_features = dim, bias = self.use_bias))

            # used for next pipe
            last_dim = dim

        
        print(self.layers)
        pass



    def forward(self, x, mask):

        out = x
        # in encoder the query, key, value are same.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # batch size, seq_len, in_feature
    dummy_query = torch.Tensor(64, 400, 2051)

    reduced_dims = [2048 // 2, 2048 // 4, 2048 // 8, 3]

    mle = MultiLayerEncoder(embed_size = 2051, reduced_dims = reduced_dims, device = device)

    out = mle(dummy_query)

    # test kqv splitting

    sa = SelfAttention(embed_size = 2051, heads = 4)

    print(sa)

    nnsa = nn.MultiheadAttention(2051 * 4, 4)

    print(nnsa)

    # 4 x 3
    mat = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 1]])

    #mat = torch.unsqueeze(mat, 0)

    drop = 0.4

    sd = nn.Dropout(p = drop)

    sd_out = sd(mat * drop)


    pass
