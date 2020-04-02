import json
import torch
import torch.nn.functional as F
from torch.utils import data

NO_SENSE = 'no_sense'


class Vocab:
    def __init__(self):
        self.index = {}
        self.inverted_index = {}
        self.running_id = 1

    def gen_id(self, string_key):
        idx = self.index.get(string_key, None)
        if not idx:
            idx = self.running_id
            self.index[string_key] = idx
            self.running_id += 1
        return idx

    def to_ids(self, token_list):
        ret = []
        for t in token_list:
            ret.append(self.gen_id(t))
        return ret

    def __invert_index(self):
        self.inverted_index = {v: k for k, v in self.index.items()}
        self.inverted_index[0] = ''

    def decode(self, ids):
        if len(self.inverted_index) != len(self.index):
            self.__invert_index()

        ret = []
        for id in ids:
            ret.append(self.inverted_index[id])
        return ret

    def size(self):
        return self.running_id


def load_raw_data(sentences_file='sentences.jsonl', senses_file='senses.jsonl', S=100):
    raw_dataset = {
        'int_sentences': [],
        'str_sentences': [],
        'int_labels': [],
        'str_labels': []
    }

    y_vocab = Vocab()
    tokens_vocab = Vocab()

    with open(sentences_file, 'r') as data:
        for i, line in enumerate(data):
            if i >= S:
                break
            tokenized_sentence = json.loads(line)
            raw_dataset['str_sentences'].append(tokenized_sentence)
            int_sentence = tokens_vocab.to_ids(tokenized_sentence)
            raw_dataset['int_sentences'].append(int_sentence)

    with open(senses_file, 'r') as data:
        for i, line in enumerate(data):
            if i >= S:
                break
            labels = json.loads(line)
            raw_dataset['str_labels'].append(labels)
            int_labels = y_vocab.to_ids(labels)
            raw_dataset['int_labels'].append(int_labels)

    return raw_dataset, tokens_vocab, y_vocab


# --> Can sort samples to create more compact batches (and then shuffle only inter-batch)
# Check lazy loading
class WSDDataset(data.Dataset):

    def __init__(self, raw_dataset, tokens_vocab, y_vocab):
        self.raw_dataset = raw_dataset

        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab

        # N holds max sentence length
        self.N = max(map(len, raw_dataset['int_sentences']))

        int_labels = raw_dataset['int_labels']
        idx_tuples = []
        for s_idx, s in enumerate(raw_dataset['int_sentences']):
            for q in range(len(s)):
                idx_tuples.append((s_idx, q, int_labels[s_idx][q]))

        self.idx_tuples = idx_tuples

    def __len__(self):
        return len(self.idx_tuples)

    def __getitem__(self, index):
        sent_idx, q, lbl = self.idx_tuples[index]
        sentence = self.raw_dataset['int_sentences'][sent_idx]

        pad = self.N - len(sentence)
        sentence_tensor = torch.tensor(sentence)
        sentence_tensor = F.pad(sentence_tensor, (0, pad))

        return sentence_tensor, torch.tensor(q), torch.tensor(lbl)

    def show_sample(self, index):
        sent_idx, q, lbl = self.idx_tuples[index]
        sentence = self.raw_dataset['int_sentences'][sent_idx]

        str_sentence = [self.tokens_vocab.index[t] for t in sentence]
        str_lbl = self.y_vocab.index[lbl]
        return str_sentence, q, lbl

    def __repr__(self):
        S = len(self.raw_dataset['int_labels'])
        return f'Samples: {len(self.idx_tuples)}\nSentences: {S} (N={self.N})\n' \
               f'Vocab:\n\tTokens:{self.tokens_vocab.size()}\n\tSenses:{self.y_vocab.size()}'



from torch import nn


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, Q, X):
        """
        Args:
            Q (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            X (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        B, output_len, D = Q.size()
        N = X.size(1)

        if self.attention_type == "general":
            Q = Q.reshape(B * output_len, D)
            Q = self.linear_in(Q)
            Q = Q.reshape(B, output_len, D)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(Q, X.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(B * output_len, N)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(B, output_len, N)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, X)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, Q), dim=2)
        combined = combined.view(B * output_len, 2 * D)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(B, output_len, D)
        output = self.tanh(output)

        return output, attention_weights