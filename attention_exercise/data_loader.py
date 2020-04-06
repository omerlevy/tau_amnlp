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
        self.padding_idx = 0
        self.running_id = 1
        self.index = {}
        self.inverted_index = {}

    def __invert_index(self):
        self.inverted_index = {v: k for k, v in self.index.items()}

    def size(self):
        return self.running_id + 1

    def encode(self, string, generate_id=True):
        """
        Get or generate and index the integer id for the given string

        :param string: string to get / generate integer id for
        :param generate_id: Whether to generate an id if string is not already indexed
        :return:
        """
        idx = self.index.get(string, None)
        if idx is None:
            if generate_id:
                idx = self.running_id
                self.index[string] = idx
                self.running_id += 1
            else:
                raise ValueError(f"{string} not found in index")
        return idx

    def encode_list(self, strings, generate_id=True):
        return [self.encode(s, generate_id=generate_id) for s in strings]

    def decode(self, id):
        """
        :param id: integer id
        :return: string representation of given id
        """
        if len(self.inverted_index) != len(self.index):
            self.__invert_index()

        if id == self.padding_idx:
            return ''
        else:
            return self.inverted_index[id]

    def decode_list(self, ids):
        return [self.decode(i) for i in ids]


def load(dataset_types, sentence_count=None):
    """
    Downloads (if necessary) the datasets, vocabularizes and loads the requested types.
    :param dataset_types: list which is a subset of ['train', 'dev', 'test']
    :param sentence_count: load only the first # senetnces
    :return:
        datasets: a dict containing the datasets
        tokens_vocab: tokens vocabulary
        y_vocab: senses vocabulary
    """
    y_vocab = Vocab()
    tokens_vocab = Vocab()

    raw_datasets_dict = {}
    for dataset_type in dataset_types:
        sentences_file = f'./data/sentences.{dataset_type}.jsonl'
        senses_file = f'./data/senses.{dataset_type}.jsonl'

        if not os.path.exists(sentences_file):
            print("Downloading tau_amnlp_semcor_dataset.zip...")
            urllib.request.urlretrieve(
                'https://docs.google.com/uc?export=download&id=1mxPS3neImDAq7BeTTQZRT4afpPU7gOge',
                "tau_amnlp_semcor_dataset.zip")
            with zipfile.ZipFile('tau_amnlp_semcor_dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('./data')
            print('Dataset was downloaded successfully and extracted to ./data')

        raw_dataset = __load_from_files(
            sentences_file, senses_file, tokens_vocab, y_vocab, sentence_count=sentence_count, include_no_sense=True)

        raw_datasets_dict[dataset_type] = raw_dataset

    wsd_datasets = {k: WSDDataset(raw_datasets_dict[k], tokens_vocab, y_vocab) for k in raw_datasets_dict}

    return wsd_datasets, tokens_vocab, y_vocab


def __load_from_files(sentences_file, senses_file, tokens_vocab, y_vocab, sentence_count=None, include_no_sense=True):
    raw_dataset = {
        'int_sentences': [],
        'str_sentences': [],
        'int_labels': [],
        'str_labels': []
    }

    with open(sentences_file, 'r') as data:
        for i, line in enumerate(data):
            if sentence_count is not None and i >= sentence_count:
                break
            tokenized_sentence = json.loads(line)
            raw_dataset['str_sentences'].append(tokenized_sentence)

    with open(senses_file, 'r') as data:
        for i, line in enumerate(data):
            if sentence_count is not None and i >= sentence_count:
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

        int_labels = y_vocab.encode_list(lbls)
        raw_dataset['int_labels'].append(int_labels)
        int_sentence = tokens_vocab.encode_list(str_sentence)
        raw_dataset['int_sentences'].append(int_sentence)

    return raw_dataset


class WSDDataset(data.Dataset):
    """
    An implementation of torch dataset in which a sample is single annotated query word.
    """

    def __init__(self, raw_dataset, tokens_vocab, y_vocab, include_no_sense=False):
        """
        :param raw_dataset: the dataset obtained from dictionary entry of the load finction
        :param tokens_vocab:
        :param y_vocab:
        :param include_no_sense: whether to consider NO_SENSE labeled words as legitimate samples
        """
        self.raw_dataset = raw_dataset
        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab
        self.include_no_sense = include_no_sense
        self.sample_type = 'word'

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


class WSDSentencesDataset(data.Dataset):

    @staticmethod
    def from_word_datasets(datasets):
        max_N = 0
        for ds in datasets.values():
            N = max(map(len, ds.raw_dataset['int_sentences']))
            if N > max_N:
                max_N = N

        new_datasets = {k: WSDSentencesDataset(
            datasets[k].raw_dataset,
            datasets[k].tokens_vocab,
            datasets[k].y_vocab,
            N=max_N
        ) for k in datasets}

        return new_datasets

    def __init__(self, raw_dataset, tokens_vocab, y_vocab, N=None):
        self.raw_dataset = raw_dataset
        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab
        self.sample_type = 'sentence'

        if N is not None:
            self.N = N
        else:
            self.N = max(map(len, raw_dataset['int_sentences']))

    def __len__(self):
        return len(self.raw_dataset['int_sentences'])

    def __getitem__(self, index):
        sentence = self.raw_dataset['int_sentences'][index]
        labels = self.raw_dataset['int_labels'][index]

        pad = self.N - len(sentence)

        sentence_tensor = F.pad(torch.tensor(sentence), (0, pad))

        labels_tensor = F.pad(torch.tensor(labels), (0, pad), value=self.y_vocab.index[NO_SENSE])

        return sentence_tensor, labels_tensor

    def __repr__(self):
        S = len(self.raw_dataset['int_labels'])
        return f'Samples: {S}\nSentences: {S} (N={self.N})\n' \
               f'Vocab:\n\tTokens:{self.tokens_vocab.size()}\n\tSenses:{self.y_vocab.size()}'