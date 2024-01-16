import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import models as models
from util.utils import *
from dataloader import load_dataset


def train(arg):
    seed_everything(arg.seed)

    train_loader, test_loader = load_dataset(batch_size=arg.batch_size)

    model = getattr(models, arg.model)().to(arg.device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=arg.lr)

    tl, ta, vl, va, cf_matrix = [], [], [], [], []
    best_epoch = 0
    best_val_acc = 0

    for epoch in range(arg.epoch):
        model.train()
        train_loss = []
        train_pred = []
        train_real = []

        for k, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(arg.device)
            pred = model(x)
            loss = criterion(pred.cpu(), y.long())
            loss.backward()
            optim.step()

            train_loss.append(loss.detach().cpu().numpy())
            train_pred.extend(list(pred.argmax(dim=1).detach().cpu().numpy()))
            train_real.extend(list(y.detach().cpu().numpy()))

        tl.append(np.mean(train_loss))
        ta.append(accuracy_score(train_real, train_pred))
        print(f'[   Train    | {epoch + 1: 03d} / {arg.epoch: 03d} ] loss = {tl[-1]: .5f}, acc = {ta[-1]:.5f}')

        model.eval()
        val_loss = []
        val_pred = []
        val_real = []

        with torch.no_grad():
            for k, batch in enumerate(test_loader):
                x, y = batch
                outputs = model(x)
                loss = criterion(outputs, y.long())
                val_loss.append(loss.detach().cpu().numpy())
                val_pred.extend(list(outputs.argmax(dim=1).detach().cpu().numpy()))
                val_real.extend(list(y.detach().cpu().numpy()))

        vl.append(np.mean(val_loss))
        va.append(accuracy_score(val_real, val_pred))
        if best_val_acc < va[-1]:
            best_val_acc = va[-1]
            best_epoch = epoch + 1

        cf_matrix = confusion_matrix(val_real, val_pred)
        print(f'[   Valid    | {epoch + 1: 03d} / {arg.epoch: 03d} ] loss = {vl[-1]: .5f}, acc = {va[-1]:.5f}')

    print(f'Best Val Acc: {best_val_acc:.5f} in {best_epoch} epochs')
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3])
    fig, axes = plt.subplots(2)
    axes[0].plot(ta, 'k')
    axes[0].plot(va, 'g--')
    axes[1].plot(tl, 'k')
    axes[1].plot(vl, 'g--')
    plt.show()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--data path', '-dp', type=str, default='./data')
    parse.add_argument('--epoch', '-e', type=int, default=10, help='number of epoch')
    parse.add_argument('--seed', '-s', type=int, default=42, help='random seed')
    parse.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parse.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate')
    parse.add_argument('--model', '-m', type=str, default='ATCNet', choices=['ATCNet', 'EEGNet'])
    parse.add_argument('--device', '-d', type=str, default='cpu', choices=['cuda', 'cpu'])

    config = parse.parse_args()

    train(config)
