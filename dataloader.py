import scipy.io as sio
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(batch_size):
    data_path = "D:/winter_vacation/Epileptic-EEG/data/"

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

    return train_loader, test_loader
