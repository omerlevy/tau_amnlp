import torch
import torch.nn as nn

import numpy as np

import pandas as pd


def evaluate(model, data_generator, iter_lim=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    correct = 0
    total = 0

    for i, (M_s, v_q, y_true) in enumerate(data_generator):
        if iter_lim is not None and i >= iter_lim:
            break

        M_s, v_q, y_true = M_s.to(device), v_q.to(device), y_true.to(device)
        y_logits, A = model(M_s, v_q)
        _, y_pred = torch.max(y_logits.data, -1)

        correct += (y_pred == y_true).sum()
        total += y_true.size(0)

    return float(correct) / float(total)


def evaluate_verbose(model, data_generator, tokens_vocab, y_vocab, iter_lim=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    correct = 0
    total = 0

    sentence_rows = []
    As = []
    v_qs = []
    y_preds = []
    y_trues = []
    y_pred_probs = []
    y_pred_losss = []

    for i, (M_s, v_q, y_true) in enumerate(data_generator):
        if iter_lim is not None and i >= iter_lim:
            break

        M_s, v_q, y_true = M_s.to(device), v_q.to(device), y_true.to(device)
        y_logits, A = model(M_s, v_q)
        _, y_pred = torch.max(y_logits.data, 1)

        correct += (y_pred == y_true).sum()
        total += y_true.size(0)

        M_s = M_s.numpy()
        for j in range(M_s.shape[0]):
            sentence_rows.append(tokens_vocab.decode(M_s[j]))

        v_qs.append(v_q.numpy())
        y_trues.append(y_true.numpy())
        y_preds.append(y_pred.numpy())
        As.append(A.detach().numpy())

        probs = nn.Softmax(dim=1)(y_logits).detach().numpy()
        y_probs = probs[range(len(probs)), y_true]
        y_pred_probs.append(y_probs)
        y_pred_losss.append(-np.log(y_probs + 1e-7))

    acc = float(correct) / float(total)

    eval_df = pd.DataFrame(sentence_rows)
    eval_df.insert(0, 'query', np.concatenate(v_qs))
    eval_df.insert(1, 'y_true', y_vocab.decode(np.concatenate(y_trues)))
    eval_df.insert(2, 'y_pred', y_vocab.decode(np.concatenate(y_preds)))
    eval_df.insert(3, 'y_pred_prob', np.concatenate(y_pred_probs))
    eval_df.insert(4, 'y_pred_loss', np.concatenate(y_pred_losss))

    eval_df.reset_index(inplace=True)

    attention_df = pd.DataFrame(np.concatenate(As))
    return acc, eval_df, attention_df


def fancy_display(eval_df, attention_df, slicer):
    I = 6
    from pandas.io.formats.style import Styler

    def highlight_sentence(row):
        q = row['query']
        style_vec = ['']*I
        style_vec += Styler._background_gradient(attention_df.loc[row['index']])
        style_vec[I+q] = 'background-color: lightgreen'
        return style_vec

    eval_df_styled = eval_df[slicer].style\
        .format({'y_pred_prob': '{:.3e}'})\
        .apply(highlight_sentence, axis=1)\
        .background_gradient(subset=['y_pred_loss'], cmap='OrRd', high=0.1)

    att_styled = attention_df[slicer].style \
        .format('{:.2e}') \
        .background_gradient(axis=1, cmap='PuBu', high=0.1)

    return eval_df_styled, att_styled