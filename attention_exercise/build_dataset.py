import json
import os

import nltk
from tqdm import tqdm

import numpy as np

from nltk.corpus import semcor

NO_SENSE = 'no_sense'


nltk.download('semcor')


def parse(tagged_sentence):
    sentence = []
    labels = []

    def append(chunk, sense):
        for s in chunk:
            sentence.append(s)
            labels.append(sense)

    for chunk in tagged_sentence:
        if type(chunk) == list:
            append(chunk, NO_SENSE)
        elif type(chunk) == nltk.tree.Tree:
            labeled_words = chunk.leaves()

            if len(labeled_words) >= 1:
                if type(chunk.label()) == nltk.corpus.reader.wordnet.Lemma:
                    # nltk 3.4.5
                    str_lbl = chunk.label().synset().name()  # + '_' + chunk.label().name()
                elif type(chunk.label()) == str:
                    str_lbl = chunk.label()
                else:
                    raise ValueError(f'Illegal chunk label {chunk.label()}')
                append(labeled_words, str_lbl)
    return sentence, labels


os.mkdir('./data')


def write_out(dataset_type, sentences, senses, idxs):
    with open(f'./data/sentences.{dataset_type}.jsonl', 'w') as sent_out:
        with open(f'./data/senses.{dataset_type}.jsonl', 'w') as sense_out:
            for i in idxs:
                json.dump(sentences[i], sent_out)
                sent_out.write('\n')

                json.dump(senses[i], sense_out)
                sense_out.write('\n')


sentences = []
senses = []

for i, ts in enumerate(tqdm(semcor.tagged_sents(tag='sem'))):
    sentence, sense = parse(ts)
    sentences.append(sentence)
    senses.append(sense)

np.random.seed(123)
sentence_idxs = np.random.permutation(len(sentences))

TRAIN_FRAC = .8
DEV_FRAC = .1

training_size = int(TRAIN_FRAC*len(sentence_idxs))
dev_size = int(DEV_FRAC*len(sentence_idxs))

write_out('train', sentences, senses, sentence_idxs[:training_size])
write_out('dev', sentences, senses, sentence_idxs[training_size:training_size + dev_size])
write_out('test', sentences, senses, sentence_idxs[training_size + dev_size:])