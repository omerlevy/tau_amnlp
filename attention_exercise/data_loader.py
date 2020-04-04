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
        # TODO: Check why this is needed (google collab problem)
        if len(string_key) == 0:
            return 0

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


def load(dataset_types, S=None, include_no_sense=True):
    raw_datasets_dict, tokens_vocab, y_vocab = load_raw_data(dataset_types, S, include_no_sense)

    ret = {}
    for k in raw_datasets_dict:
        ret[k] = WSDDataset(raw_datasets_dict[k], tokens_vocab, y_vocab)

    return ret, tokens_vocab, y_vocab


def load_raw_data(dataset_types, S=None, include_no_sense=True):
    y_vocab = Vocab()
    tokens_vocab = Vocab()

    ret = {}
    for dataset_type in dataset_types:
        sentences_file = f'./data/sentences.{dataset_type}.jsonl'
        senses_file = f'./data/senses.{dataset_type}.jsonl'

        if not os.path.exists(sentences_file):
            print("Downloading tau_amnlp_semcor_dataset.zip...")
            # https://drive.google.com/file/d/1mxPS3neImDAq7BeTTQZRT4afpPU7gOge/view?usp=sharing
            urllib.request.urlretrieve(
                'https://docs.google.com/uc?export=download&id=1mxPS3neImDAq7BeTTQZRT4afpPU7gOge',
                "tau_amnlp_semcor_dataset.zip")
            with zipfile.ZipFile('tau_amnlp_semcor_dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('./data')
            print('Dataset was downloaded successfully and extracted to ./data')

        raw_dataset = __load_from_files(
            sentences_file, senses_file, tokens_vocab, y_vocab, S=S, include_no_sense=include_no_sense)

        ret[dataset_type] = raw_dataset

    return ret, tokens_vocab, y_vocab


def __load_from_files(sentences_file, senses_file, tokens_vocab, y_vocab, S=None, include_no_sense=True):
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

    with open(senses_file, 'r') as data:
        for i, line in enumerate(data):
            if S is not None and i >= S:
                break
            labels = json.loads(line)
            raw_dataset['str_labels'].append(labels)

    for i, lbls in enumerate(raw_dataset['str_labels']):
        str_sentence = raw_dataset['str_sentences'][i]
        if not include_no_sense:
            sense_idxs = [j for j in range(len(lbls)) if lbls[j] != NO_SENSE]
            # print(sense_idxs)
            lbls = [lbls[j] for j in sense_idxs]
            str_sentence = [str_sentence[j] for j in sense_idxs]
            raw_dataset['str_sentences'][i] = str_sentence
            raw_dataset['str_labels'][i] = lbls

        int_labels = y_vocab.to_ids(lbls)
        raw_dataset['int_labels'].append(int_labels)
        int_sentence = tokens_vocab.to_ids(str_sentence)
        raw_dataset['int_sentences'].append(int_sentence)

    return raw_dataset


class WSDDataset(data.Dataset):

    def __init__(self, raw_dataset, tokens_vocab, y_vocab, include_no_sense=False):
        self.raw_dataset = raw_dataset
        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab
        self.include_no_sense = include_no_sense

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


class WSDSelfAttentionDataset(data.Dataset):

    def __init__(self, raw_dataset, tokens_vocab, y_vocab, include_no_sense=False):
        self.raw_dataset = raw_dataset
        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab
        self.include_no_sense = include_no_sense

        self.sentence_idxs = []
        self.queries = None
        self.labels = None

        if include_no_sense:
            self.qN = self.N
            q_ranges = map(range, raw_dataset['int_sentences'])
            self.queries = list(map(list, q_ranges))
            self.labels = raw_dataset['int_labels']
        else:
            self.queries = []
            self.labels = []
            for s_idx, s in enumerate(raw_dataset['int_sentences']):
                sense_query_idxs = [q for q in range(len(s)) if raw_dataset['str_labels'][s_idx][q] != NO_SENSE]
                self.queries.append(sense_query_idxs)
                self.labels.append([raw_dataset['int_labels'][s_idx][j] for j in sense_query_idxs])

        self.sentence_idxs = [sidx for sidx in range(len(raw_dataset['int_sentences'])) if len(self.labels[sidx]) > 0]
        # N holds max sentence length
        self.N = max(map(len, raw_dataset['int_sentences']))
        # qN holds max query length
        self.qN = max(map(len, self.queries))
        # print('x')

    def __len__(self):
        return len(self.sentence_idxs)

    def __getitem__(self, index):
        # print(index)
        index = self.sentence_idxs[index]
        sentence = self.raw_dataset['int_sentences'][index]
        queries = self.queries[index]
        labels = self.labels[index]

        sent_pad = self.N - len(sentence)
        q_pad = self.qN - len(queries)

        sentence_tensor = F.pad(torch.tensor(sentence), (0, sent_pad))

        queries_tensor = F.pad(torch.tensor(queries), (0, q_pad))

        labels_tensor = F.pad(torch.tensor(labels), (0, q_pad))

        # print(queries_tensor.dtype)

        return sentence_tensor, queries_tensor, labels_tensor

    def __repr__(self):
        S = len(self.raw_dataset['int_labels'])
        return f'Samples: {S}\nSentences: {S} (N={self.N})\n' \
               f'Vocab:\n\tTokens:{self.tokens_vocab.size()}\n\tSenses:{self.y_vocab.size()}'
