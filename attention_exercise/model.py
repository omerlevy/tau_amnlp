import torch
from torch import nn
import torch.nn.init as init
import math


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class WSDModel(nn.Module):
    def __init__(self, V, Y, D=300, dropout_prob=0.2, q_dim=1):
        super(WSDModel, self).__init__()
        self.D = D

        self.E_v = nn.Embedding(V, D, padding_idx=0)
        self.E_y = nn.Embedding(Y, D, padding_idx=0)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))
        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([q_dim, self.D])

    def attention(self, X, Q):
        # X: [B, N, D]
        # Q: [B, k, D] with k=1 or N

        Q = Q @ self.W_A / math.sqrt(self.D)

        # A: [B, 1, N]
        A = self.softmax(Q @ X.transpose(1, 2))
        Q_c = A @ X

        Q_c = Q_c @ self.W_O

        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None):
        # M_s: [B, N]
        # v_q: [B]

        X = self.dropout_layer(self.E_v(M_s))

        if v_q is not None:
            # Broadcast v_q to [B, 1, D] for gather
            broadcast_shape = (v_q.shape[0], 1, self.D)
            broadcast_tensor = torch.ones(broadcast_shape, dtype=int, device=device)
            M_q_broadcasted = v_q[:, None, None] * broadcast_tensor

            # [B, 1, D]
            Q = torch.gather(X, 1, M_q_broadcasted)
        else:
            Q = X

        Q_c, A = self.attention(X, Q)

        H = self.layer_norm(Q_c + Q)

        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A
