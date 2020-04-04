import torch
from torch import nn
import torch.nn.init as init
import math


class WSDModel(nn.Module):
    def __init__(self, V, Y, D=300, dropout_prob=0.2):
        super(WSDModel, self).__init__()
        self.D = D

        self.E_v = nn.Embedding(V, D)
        self.E_y = nn.Embedding(Y, D)

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))

        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        # init.kaiming_normal_(self.W_A)
        # init.kaiming_normal_(self.W_O)

        #         init.normal_(self.W_A)
        #         init.normal_(self.W_O)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([1, self.D])

    def attention(self, X, Q):
        # X: [B, N, D]
        # Q: [B, 1, D]

        Q = Q @ self.W_A / math.sqrt(self.D)

        # A: [B, 1, N]
        A = self.softmax(Q @ X.transpose(1, 2))
        Q_c = A @ X

        Q_c = Q_c @ self.W_O

        return Q_c, A

    def forward(self, M_s, v_q):
        # M_s: [B, N]
        # v_q: [B]

        X = self.dropout_layer(self.E_v(M_s))

        # TODO: https://pytorch.org/docs/stable/torch.html#torch.gather
        Q_idxs = M_s[range(v_q.shape[0]), v_q]
        Q = self.dropout_layer(self.E_v(Q_idxs).unsqueeze(1))

        Q_c, A = self.attention(X, Q)

        H = self.layer_norm(Q_c + Q)

        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A.squeeze()