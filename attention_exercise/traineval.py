import torch
from tqdm import tqdm
from torch import nn
from torch.utils import data
import numpy as np
import pandas as pd

NO_SENSE = 'no_sense'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(model, optimizer, wsd_train_dataset, wsd_dev_dataset, num_epochs=20, batch_size=100):
    """
    Run the training loop for the given arguments.

    :param model: Attention model, instance of torch.nn.Module
    :param optimizer: An instance of @orch.optim.Optimizer
    :param wsd_train_dataset: The train dataset
    :param wsd_dev_dataset: The dev dataset
    :param num_epochs: Number of epochs to train
    :param batch_size: Batch size
    :return:
        train_loss: training loss for each iteration
        train_acc: training accuracy sample per epoch
        val_acc: dev accuracy per epoch
    """
    is_sentence_samples = wsd_train_dataset.sample_type == 'sentence'

    training_generator = data.DataLoader(
        wsd_train_dataset, batch_size=batch_size, shuffle=True
    )

    val_generator = data.DataLoader(
        wsd_dev_dataset, batch_size=batch_size, shuffle=True
    )

    no_sense_index = wsd_train_dataset.y_vocab.index[NO_SENSE]
    ce_loss = nn.CrossEntropyLoss(ignore_index=no_sense_index)

    train_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        model.eval()
        cur_train_acc = evaluate(model, training_generator, iter_lim=100, ignore_label=no_sense_index)
        train_acc.append(cur_train_acc)

        cur_val_acc = evaluate(model, val_generator, ignore_label=no_sense_index)
        val_acc.append(cur_val_acc)

        model.train()
        with tqdm(training_generator) as prg_train:
            for i, sample in enumerate(prg_train):
                optimizer.zero_grad()

                y_logits, y_true = __run_sample(sample, model, is_sentence_samples)

                loss = ce_loss(y_logits, y_true)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                status_str = f'[{epoch}] loss: {train_loss[-1]:.3f}'
                prg_train.set_description(status_str)

    return train_loss, train_acc, val_acc


def evaluate(model, data_generator, iter_lim=None, ignore_label=None):
    """
    Evaluate the given model on the given dataset.

    :param model: Attention model, instance of torch.nn.Module
    :param data_generator: Instance of torch.utils.data.DataLoader
    :param iter_lim: Run only the specified iterations count
    :param ignore_label: If not None, ignores the given label in accuracy computation
    :return: Accuracy; correct/total
    """
    is_sentence_samples = data_generator.dataset.sample_type == 'sentence'

    with torch.set_grad_enabled(False):

        correct = 0
        total = 0

        for i, sample in enumerate(data_generator):
            if iter_lim is not None and i >= iter_lim:
                break

            y_logits, y_true = __run_sample(sample, model, is_sentence_samples)

            _, y_pred = torch.max(y_logits.data, -1)

            if ignore_label is not None:
                sense_mask = y_true.ne(ignore_label)

                y_true = y_true[sense_mask]
                y_pred = y_pred[sense_mask]

            correct += (y_pred == y_true).sum()
            total += y_true.size(0)

        return float(correct) / float(total)


def __run_sample(sample, model, is_sentence_samples):
    if is_sentence_samples:
        M_s, y_true = sample
        M_s, y_true = M_s.to(device), y_true.to(device)
        y_logits, A = model(M_s)
        flat_y_logits = y_logits.reshape([-1, y_logits.shape[2]])
        flat_y_true = y_true.reshape(-1)
        return flat_y_logits, flat_y_true
    else:
        M_s, v_q, y_true = sample
        M_s, v_q, y_true = M_s.to(device), v_q.to(device), y_true.to(device)
        y_logits, A = model(M_s, v_q)
        return y_logits, y_true


def evaluate_verbose(model, dataset, iter_lim=10, shuffle=True):

    g = data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=shuffle,
    )

    tokens_vocab = dataset.tokens_vocab
    y_vocab = dataset.y_vocab

    correct = 0
    total = 0

    sentence_rows = []
    q_tokens = []
    As = []
    v_qs = []
    y_preds = []
    y_trues = []
    y_pred_probs = []
    y_pred_losss = []

    for i, (M_s, v_q, y_true) in enumerate(g):
        if iter_lim is not None and i >= iter_lim:
            break

        M_s, v_q, y_true = M_s.to(device), v_q.to(device), y_true.to(device)
        y_logits, A = model(M_s, v_q)
        _, y_pred = torch.max(y_logits.data, 1)

        correct += (y_pred == y_true).sum()
        total += y_true.size(0)

        M_s = M_s.cpu().numpy()
        for j in range(M_s.shape[0]):
            sentence_rows.append(tokens_vocab.decode_list(M_s[j]))
            q_tokens.append(tokens_vocab.inverted_index[M_s[j][v_q[j]]])

        v_qs.append(v_q.cpu().numpy())
        y_true_np = y_true.cpu().numpy()
        y_trues.append(y_true_np)
        y_preds.append(y_pred.cpu().numpy())
        As.append(A.detach().cpu().numpy())

        probs = nn.Softmax(dim=1)(y_logits).detach().cpu().numpy()
        y_probs = probs[range(len(probs)), y_true_np]
        y_pred_probs.append(y_probs)
        y_pred_losss.append(-np.log(y_probs + 1e-7))

    acc = float(correct) / float(total)

    eval_df = pd.DataFrame(sentence_rows)
    eval_df.insert(0, 'query', np.concatenate(v_qs))
    eval_df.insert(1, 'query_token', q_tokens)
    eval_df.insert(2, 'y_true', y_vocab.decode_list(np.concatenate(y_trues)))
    eval_df.insert(3, 'y_pred', y_vocab.decode_list(np.concatenate(y_preds)))
    eval_df.insert(4, 'y_pred_prob', np.concatenate(y_pred_probs))
    eval_df.insert(5, 'y_pred_loss', np.concatenate(y_pred_losss))

    eval_df.reset_index(inplace=True)

    attention_df = pd.DataFrame(np.concatenate(As))
    return acc, eval_df, attention_df


def highlight(eval_df, attention_df, slicer):
    I = 7
    from pandas.io.formats.style import Styler

    def highlight_sentence(row):
        q = row['query']
        style_vec = [''] * I
        style_vec += Styler._background_gradient(attention_df.loc[row['index']])
        style_vec[I + q] = 'background-color: lightgreen'
        return style_vec

    eval_df_styled = eval_df.iloc[slicer].style \
        .format({'y_pred_prob': '{:.3e}'}) \
        .apply(highlight_sentence, axis=1) \
        .background_gradient(subset=['y_pred_loss'], cmap='OrRd', high=0.1)

    att_styled = attention_df.iloc[slicer].style \
        .format('{:.2e}') \
        .background_gradient(axis=1, cmap='PuBu', high=0.1)

    return eval_df_styled, att_styled
