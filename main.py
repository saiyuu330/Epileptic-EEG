from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from models.ATCNET import *


def train():
    data_path = "D:/winter_vacation/Epileptic-EEG/data/"
    epochs = 50
    batch_size = 128
    lr = 1e-04
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    model = ATCNet().to(torch.float)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = sio.loadmat(data_path + "x_train.mat")
    train_labels = sio.loadmat(data_path + "y_train.mat")
    test_data = sio.loadmat(data_path + "x_test.mat")
    test_labels = sio.loadmat(data_path + "y_test.mat")

    train_data = train_data["x_train"]
    train_labels = train_labels["y_train"][0]
    test_data = test_data["x_test"]
    test_labels = test_labels["y_test"][0]

    train_tensor_data = torch.tensor(train_data, dtype=torch.float)
    train_tensor_data = train_tensor_data.unsqueeze(1)
    train_tensor_labels = torch.tensor(train_labels)
    test_tensor_data = torch.tensor(test_data, dtype=torch.float)
    test_tensor_data = test_tensor_data.unsqueeze(1)
    test_tensor_labels = torch.tensor(test_labels)

    train_dataset = TensorDataset(train_tensor_data, train_tensor_labels)
    test_dataset = TensorDataset(test_tensor_data, test_tensor_labels)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    tl, ta, vl, va = [], [], [], []
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_pred = []
        train_real = []

        for k, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            pred = model(x)
            loss = criterion(pred.cpu(), y.long())
            loss.backward()
            optim.step()

            train_loss.append(loss.detach().cpu().numpy())
            train_pred.extend(list(pred.argmax(dim=1).detach().cpu().numpy()))
            train_real.extend(list(y.detach().cpu().numpy()))

        tl.append(np.mean(train_loss))
        ta.append(accuracy_score(train_real, train_pred))
        print(f'[   Train    | {epoch + 1: 03d} / {epochs: 03d} ] loss = {tl[-1]: .5f}, acc = {ta[-1]:.5f}')

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
        print(f'[   Valid    | {epoch + 1: 03d} / {epochs: 03d} ] loss = {vl[-1]: .5f}, acc = {va[-1]:.5f}')

    fig, axes = plt.subplots(2)
    axes[0].plot(ta, color='k*')
    axes[0].plot(va, color='g*')
    axes[1].plot(tl, color='k--')
    axes[1].plot(vl, color='g--')
    plt.show()


if __name__ == "__main__":
    train()
