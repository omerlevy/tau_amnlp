import torch
from tqdm import tqdm
import statistics
from torch import nn
from torch.utils import data


def train(model, optimizer, wsd_train_dataset, wsd_dev_dataset, num_epochs=20, batch_size=100):

    training_generator = data.DataLoader(
        wsd_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    val_generator = data.DataLoader(
        wsd_dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ce_loss = nn.CrossEntropyLoss()

    losses = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        # print(f"lr = {optimizer.param_groups[0]['lr']}")
        model.train()
        with tqdm(training_generator) as prg_train:
            for i, (M_s, v_q, y_true) in enumerate(prg_train):
                M_s, v_q, y_true = M_s.to(device), v_q.to(device), y_true.to(device)

                ## SHAPES:
                # M_s     --> [B, N]
                # M_q     --> [B]
                # y_true  --> [B]

                optimizer.zero_grad()

                y_logits, _ = model(M_s, v_q)
                #             print(y_logits.shape)
                loss = ce_loss(y_logits, y_true)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                running_mean_loss = statistics.mean(losses[-min(len(losses), 100):])
                status_str = f'[{epoch}] loss: {running_mean_loss:.3f}'
                prg_train.set_description(status_str)

            model.eval()
            cur_train_acc = evaluate(model, training_generator, iter_lim=100)
            train_acc.append(cur_train_acc)

            cur_val_acc = evaluate(model, val_generator)
            val_acc.append(cur_val_acc)
            # prg_train.set_description(f'[{epoch}] loss: {running_mean_loss:.3f}, train_acc:{cur_train_acc:.3f}, val_acc:{cur_val_acc:.3f}')
            # print(f'train_acc:{cur_train_acc:.3f}, val_acc:{cur_val_acc:.3f}')

    return losses, train_acc, val_acc


def evaluate(model, data_generator, iter_lim=None):
    with torch.set_grad_enabled(False):
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