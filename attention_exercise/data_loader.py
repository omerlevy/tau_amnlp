import numpy as np
import torch
from torch.utils import data
import nltk
from nltk.corpus import semcor
import re


NO_SENSE = 'no_sense'

RE_NON_CHARS = re.compile('[^a-zA-Z]')


# Questions
# 1) trim panctuation?
# 2) What to do about no_sense?
# 3) Single sample per query word is appropriate?


def trim(words):
    ret = []
    for w in words:
        w = RE_NON_CHARS.sub('', w).strip()
        if len(w) > 0:
            ret.append(w)
    return ret


class Vocab:
    def __init__(self):
        self.index = {}
        self.running_id = 0

    def gen_id(self, string_key):
        idx = self.index.get(string_key, None)
        if not idx:
            idx = self.running_id
            self.index[string_key] = idx
            self.running_id += 1
        return idx

    def size(self):
        return self.running_id


def load_raw_data(S=100):
    raw_dataset = {
        'int_sentences': [],
        'str_sentences': [],
        'int_labels': [],
        'str_labels': [],
        'queries': [],
        'labels': []
    }

    y_vocab = Vocab()
    tokens_vocab = Vocab()

    def parse(tagged_sentence):
        int_sentence = []
        str_sentence = []
        int_labels = []
        str_labels = []

        def append(chunk, sense):
            sense_id = y_vocab.gen_id(sense)
            for s in chunk:
                int_sentence.append(tokens_vocab.gen_id(s))
                str_sentence.append(s)
                int_labels.append(sense_id)
                str_labels.append(sense)

        for chunk in tagged_sentence:
            if type(chunk) == list:
                append(trim(chunk), NO_SENSE)
            elif type(chunk) == nltk.tree.Tree:
                labeled_words = trim(chunk.leaves())

                if len(labeled_words) >= 1:
                    if type(chunk.label()) == str:
                        # nltk 3.2.5
                        str_lbl = chunk.label()
                    elif type(chunk.label()) == nltk.corpus.reader.wordnet.Lemma:
                        # nltk 3.4.5
                        str_lbl = chunk.label().synset().name() + '_' + chunk.label().name()
                    else:
                        raise ValueError(f'Illegal chunk label {chunk.label()}')
                    append(labeled_words, str_lbl)

        qs = []
        lbls = []
        for q_sentence_idx, lbl in enumerate(int_labels):
            if lbl is not None:
                qs.append(int_sentence[q_sentence_idx])
                lbls.append(lbl)

        if len(lbls) > 0:
            raw_dataset['int_sentences'].append(int_sentence)
            raw_dataset['str_sentences'].append(str_sentence)
            raw_dataset['int_labels'].append(int_labels)
            raw_dataset['str_labels'].append(str_labels)

            raw_dataset['queries'].append(qs)
            raw_dataset['labels'].append(lbls)

    nltk.download('semcor')

    for i, ts in enumerate(semcor.tagged_sents(tag='sem')[:S]):
        parse(ts)

    sample_idx_tuples = []
    for i, s in enumerate(raw_dataset['int_sentences']):
        qs = raw_dataset['queries'][i]
        lbls = raw_dataset['labels'][i]
        for qi, q in enumerate(qs):
            sample_idx_tuples.append((i, q, lbls[qi]))

    raw_dataset['idx_tuples'] = sample_idx_tuples
    return raw_dataset, tokens_vocab, y_vocab


class WSDDataset(data.Dataset):

    def __init__(self, raw_dataset, V, sample_idxs=None, N=200):
        self.raw_dataset = raw_dataset

        if sample_idxs is None:
            sample_idxs = raw_dataset['idx_tuples'].copy()

        self.sample_idxs = sample_idxs
        self.V = V
        self.N = N

    def __len__(self):
        return len(self.sample_idxs)

    def sentence_to_mat(self, int_sentence, force_rows=None):
        S = len(int_sentence)
        if force_rows and S < force_rows:
            S = force_rows
        M = np.zeros([S, self.V])
        M[range(len(int_sentence)), int_sentence] = 1
        return torch.tensor(M, dtype=torch.float32)

    def gen_sample(self, sentence, q, lbl):
        M_s = self.sentence_to_mat(sentence, force_rows=self.N)
        v_q = self.sentence_to_mat([q])
        return M_s, v_q, torch.tensor(lbl)

    def __getitem__(self, index):
        sent_idx, q, lbl = self.sample_idxs[index]
        sentence = self.raw_dataset['int_sentences'][sent_idx]

        M_s, v_q, y_true = self.gen_sample(sentence, q, lbl)

        return M_s, v_q, y_true

    def gen_batches_idxs(self, batch_size=32):
        idx_tuples = self.raw_dataset['idx_tuples'].copy()
        np.random.shuffle(idx_tuples)
        batches = np.array_split(idx_tuples, len(idx_tuples) / batch_size)
        return batches

    def iterate_batches(self, batches, N=100):
        def iter_samples(sample_idx_tuples):
            for sent_idx, q, lbl in sample_idx_tuples:
                sentence = self.raw_dataset['int_sentences'][sent_idx]
                yield self.gen_sample(sentence, q, lbl)

        for b in batches:
            M_ss_l = []
            M_qs_l = []
            y_trues = []
            for M_s, M_q, lbl in iter_samples(b):
                M_ss_l.append(M_s)
                M_qs_l.append(M_q)
                y_trues.append(lbl)
            M_ss = torch.stack(M_ss_l)
            M_qs = torch.stack(M_qs_l)
            y_true = torch.stack(y_trues)
            yield M_ss, M_qs, y_true