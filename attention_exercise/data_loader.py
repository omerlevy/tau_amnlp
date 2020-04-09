import json
import os

import torch
import torch.nn.functional as F
from torch.utils import data

import urllib
import zipfile

PAD = ''
UNK = '<UNK>'
NO_SENSE = 'no_sense'


class Vocab:

    def __init__(self):
        self.padding_idx = 0
        self.unk_idx = 1
        self.running_id = 2
        self.index = {PAD: 0, UNK: 1}
        self.inverted_index = self.__invert_index()

    def __invert_index(self):
        return {v: k for k, v in self.index.items()}

    def size(self):
        return self.running_id

    def encode(self, string, generate_id=True):
        """
        Get or generate and index the integer id for the given string.

        :param string: string to get / generate integer id for
        :param generate_id: Whether to generate an id if string is not already indexed
        :return:
        """
        idx = self.index.get(string, self.unk_idx)
        if idx is self.unk_idx:
            if generate_id:
                idx = self.running_id
                self.index[string] = idx
                self.running_id += 1
        return idx

    def encode_list(self, strings, generate_id=True):
        return [self.encode(s, generate_id=generate_id) for s in strings]

    def decode(self, id):
        """
        :param id: integer id
        :return: string representation of given id
        """
        if len(self.inverted_index) != len(self.index):
            self.inverted_index = self.__invert_index()

        return self.inverted_index[id]

    def decode_list(self, ids):
        return [self.decode(i) for i in ids]


def load_train_dataset(sentence_count=None):
    """
    Downloads (if necessary) the datasets, vocabularizes and loads the training set.
    :param sentence_count: load only the first # senetnces
    :return:
        dataset: The training set; an instance of WSDDataset
        tokens_vocab: tokens vocabulary
        y_vocab: senses vocabulary
    """
    return __load('train', sentence_count=sentence_count)


def load_dev_dataset(tokens_vocab, y_vocab, sentence_count=None):
    """
    Loads the dev set using the given vocabularies for encoding. Unknown tokens will be encoded with UNK.
    :param tokens_vocab: Tokens vocabulary obtained form training set
    :param y_vocab: Senses vocabulary obtained form training set
    :param sentence_count: load only the first # senetnces
    :return:
        dataset: The dev set; an instance of WSDDataset
    """
    return __load('dev', tokens_vocab=tokens_vocab, y_vocab=y_vocab, sentence_count=sentence_count)[0]


def load_test_dataset(tokens_vocab, y_vocab, sentence_count=None):
    """
    Loads the test set using the given vocabularies for encoding. Unknown tokens will be encoded with UNK.
    :param tokens_vocab: Tokens vocabulary obtained form training set
    :param y_vocab: Senses vocabulary obtained form training set
    :param sentence_count: load only the first # senetnces
    :return:
        dataset: The test set; an instance of WSDDataset
    """
    return __load('test', tokens_vocab=tokens_vocab, y_vocab=y_vocab, sentence_count=sentence_count)[0]


def __load(dataset_type, tokens_vocab=None, y_vocab=None, sentence_count=None):
    """
    Downloads (if necessary) the datasets, vocabularizes and loads the requested types.
    :param dataset_type: one of ['train', 'dev', 'test']
    :param sentence_count: load only the first # senetnces
    :return:
        dataset: The requested dataset; an instance of WSDDataset
        tokens_vocab: tokens vocabulary
        y_vocab: senses vocabulary
    """
    gen_vocab_ids = None
    if y_vocab is None and tokens_vocab is None:
        gen_vocab_ids = True
        y_vocab = Vocab()
        tokens_vocab = Vocab()
    elif y_vocab is not None and tokens_vocab is not None:
        gen_vocab_ids = False
    else:
        raise ValueError("Vocabularies must either be both provided or both not")

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

    raw_dataset = __load_from_files(sentences_file, senses_file, tokens_vocab, y_vocab,
                                    sentence_count=sentence_count, gen_vocab_ids=gen_vocab_ids)

    wsd_dataset = WSDDataset(raw_dataset, tokens_vocab, y_vocab)

    return wsd_dataset, tokens_vocab, y_vocab


def __load_from_files(sentences_file, senses_file, tokens_vocab, y_vocab,
                      sentence_count=None, gen_vocab_ids=True):
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
        int_labels = y_vocab.encode_list(lbls, generate_id=gen_vocab_ids)
        raw_dataset['int_labels'].append(int_labels)
        int_sentence = tokens_vocab.encode_list(str_sentence, generate_id=gen_vocab_ids)
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
    def from_word_dataset(dataset):
        return WSDSentencesDataset(
            dataset.raw_dataset,
            dataset.tokens_vocab,
            dataset.y_vocab,
        )

    def __init__(self, raw_dataset, tokens_vocab, y_vocab):
        self.raw_dataset = raw_dataset
        self.tokens_vocab = tokens_vocab
        self.y_vocab = y_vocab
        self.sample_type = 'sentence'
        self.no_sense_label = y_vocab.encode(NO_SENSE)
        self.N = max(map(len, raw_dataset['int_sentences']))

    def __len__(self):
        return len(self.raw_dataset['int_sentences'])

    def __getitem__(self, index):
        sentence = self.raw_dataset['int_sentences'][index]
        labels = self.raw_dataset['int_labels'][index]

        pad = self.N - len(sentence)
        sentence_tensor = F.pad(torch.tensor(sentence), (0, pad), value=self.tokens_vocab.padding_idx)
        labels_tensor = F.pad(torch.tensor(labels), (0, pad), value=self.no_sense_label)

        return sentence_tensor, labels_tensor

    def __repr__(self):
        S = len(self.raw_dataset['int_labels'])
        return f'Samples: {S}\nSentences: {S} (N={self.N})\n' \
               f'Vocab:\n\tTokens:{self.tokens_vocab.size()}\n\tSenses:{self.y_vocab.size()}'