import json
import os

import torch
import torch.nn.functional as F
from torch.utils import data

import urllib
import zipfile

NO_SENSE = 'no_sense'

class Vocab:
    def __init__(self):
        self.index = {
            '': 0
        }
        self.inverted_index = {
            0: ''
        }
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

    def decode(self, ids):
        if len(self.inverted_index) != len(self.index):
            self.__invert_index()

        ret = []
        for id in ids:
            ret.append(self.inverted_index[id])
        return ret

    def size(self):
        return self.running_id


def load_raw_data(S=None):
    sentences_train_file = './data/sentences.train.jsonl'
    senses_train_file = './data/senses.train.jsonl'
    sentences_test_file = './data/sentences.test.jsonl'
    senses_test_file = './data/senses.test.jsonl'

    if not os.path.exists(sentences_train_file):
        print("Downloading tau_amnlp_semcor_dataset.zip...")
        # https://drive.google.com/file/d/1bCs3xj8LHjGdof-68Opvccv4mfrXXGTH/view?usp=sharing
        # prev --> https://docs.google.com/uc?export=download&id=1VmId8L7L8QGWu7QwRHgm309xHN3O2jOM
        urllib.request.urlretrieve('https://docs.google.com/uc?export=download&id=1bCs3xj8LHjGdof-68Opvccv4mfrXXGTH',
                                   "tau_amnlp_semcor_dataset.zip")
        with zipfile.ZipFile('tau_amnlp_semcor_dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('./data')
        print('Dataset was downloaded successfully and extracted to ./data')

    y_vocab = Vocab()
    tokens_vocab = Vocab()

    raw_train_dataset = __load_from_files(sentences_train_file, senses_train_file, tokens_vocab, y_vocab, S=S)
    raw_test_dataset = __load_from_files(sentences_test_file, senses_test_file, tokens_vocab, y_vocab, S=S)

    return raw_train_dataset, raw_test_dataset, tokens_vocab, y_vocab


def __load_from_files(sentences_file, senses_file, tokens_vocab, y_vocab, S=None):
    raw_dataset = {
        'int_sentences': [],
        'str_sentences': [],
        'int_labels': [],
        'str_labels': []
    }

    with open(sentences_file, 'r') as data:
        for i, line in enumerate(data):
            if S is not None and i >= S:
                break
            tokenized_sentence = json.loads(line)
            raw_dataset['str_sentences'].append(tokenized_sentence)
            int_sentence = tokens_vocab.to_ids(tokenized_sentence)
            raw_dataset['int_sentences'].append(int_sentence)

    with open(senses_file, 'r') as data:
        for i, line in enumerate(data):
            if S is not None and i >= S:
                break
            labels = json.loads(line)
            raw_dataset['str_labels'].append(labels)
            int_labels = y_vocab.to_ids(labels)
            raw_dataset['int_labels'].append(int_labels)

    return raw_dataset


# --> Can sort samples to create more compact batches (and then shuffle only inter-batch)
# Check lazy loading
class WSDDataset(data.Dataset):

    def __init__(self, raw_dataset, tokens_vocab, y_vocab, include_no_sense=False):
        self.raw_dataset = raw_dataset

        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab

        # N holds max sentence length
        self.N = max(map(len, raw_dataset['int_sentences']))

        int_labels = raw_dataset['int_labels']
        str_labels = raw_dataset['str_labels']
        no_sense_count = 0
        idx_tuples = []
        for s_idx, s in enumerate(raw_dataset['int_sentences']):
            for q in range(len(s)):
                is_no_sense = str_labels[s_idx][q] == NO_SENSE
                if include_no_sense or not is_no_sense:
                    idx_tuples.append((s_idx, q, int_labels[s_idx][q]))
                    if is_no_sense:
                        no_sense_count += 1

        self.idx_tuples = idx_tuples
        self.no_sense_count = no_sense_count

    def __len__(self):
        return len(self.idx_tuples)

    def __getitem__(self, index):
        sent_idx, q, lbl = self.idx_tuples[index]
        sentence = self.raw_dataset['int_sentences'][sent_idx]

        pad = self.N - len(sentence)
        sentence_tensor = torch.tensor(sentence)
        sentence_tensor = F.pad(sentence_tensor, (0, pad))

        return sentence_tensor, torch.tensor(q), torch.tensor(lbl)

    def __repr__(self):
        S = len(self.raw_dataset['int_labels'])
        return f'Samples: {len(self.idx_tuples)} (no_sense: {self.no_sense_count})\nSentences: {S} (N={self.N})\n' \
               f'Vocab:\n\tTokens:{self.tokens_vocab.size()}\n\tSenses:{self.y_vocab.size()}'
