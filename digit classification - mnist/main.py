import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 10
LEARNING_RATE = 0.01
BEST_MODEL_DIR = r'model'

plt.style.use('ggplot')

if not os.path.exists(BEST_MODEL_DIR):
    os.mkdir(BEST_MODEL_DIR)


class Data:

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def generate(self):
        train_dataset = MNIST(root='./data', train=True,
                              transform=transforms.ToTensor(), download=True)
        test_dataset = MNIST(root='./data', train=False,
                             transform=transforms.ToTensor(), download=True)

        # Apply Transforms
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        train_dataset.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            normalize
        ])

        '''Data augmentation is unnecessary for test dataset'''
        test_dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader

    def random_sample(self, loader, size=1):
        data, target = next(iter(loader))
        idx = torch.randperm(len(data))[:size]

        return data[idx], target[idx]


class Model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)

        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            probabilities = torch.exp(x)
            predicted_digits = probabilities.argmax(dim=1)

            return predicted_digits.tolist()

    def visualize(self, x, y):
        y_act = self.predict(x)
        _, ax = plt.subplots(len(x) // 4, 4, figsize=(20, 10))

        for i in range(len(x)):
            ax[i // 4, i % 4].imshow(x[i].view(28, 28), cmap='gray')
            ax[i // 4, i % 4].set_title(
                f'Actual: {y[i].item() if type(y[i]) != int else y[i]}, Predicted: {y_act[i]}')
            ax[i // 4, i % 4].axis('off')

        plt.show()


class Trainer:

    def __init__(self, model, device, optimizer) -> None:
        self.model = model
        self.device = device
        self.optim = optimizer

    def train(self, train_loader, epoch):
        self.model.train()
        pbar = tqdm(train_loader)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optim.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optim.step()

            pbar.set_description(desc=f'Epoch={epoch}\tLoss={loss.item():.2f}')

        return loss.item()

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print(
                f'\nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')

        return test_loss, correct / len(test_loader.dataset) * 100


class Helper:

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def delete_files_in_path(path):
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

    @staticmethod
    def get_best_loss():
        if len(os.listdir(BEST_MODEL_DIR)) > 0:
            return float(os.listdir(BEST_MODEL_DIR)[0].split('--loss-')[1].split('--')[0])
        else:
            return 0.0

    @staticmethod
    def early_stop(test_losses, patience=3):
        if len(test_losses) < patience:
            return False
        else:
            return test_losses.index(min(test_losses)) < len(test_losses) - patience - 1

    @staticmethod
    def load_model(model, path):
        model.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def visualize(num_samples=16):
        _, test_loader = Data().generate()
        model = Model()
        if len(os.listdir(BEST_MODEL_DIR)) > 0:
            model = Helper.load_model(model, os.path.join(
                BEST_MODEL_DIR, os.listdir(BEST_MODEL_DIR)[0]))

        data = Data().random_sample(test_loader, num_samples)
        model.visualize(data[0], data[1])

    @staticmethod
    def visualize_incorrect(num_samples=16):
        _, test_loader = Data(batch_size=1).generate()
        model = Model()

        if len(os.listdir(BEST_MODEL_DIR)) > 0:
            model = Helper.load_model(model, os.path.join(
                BEST_MODEL_DIR, os.listdir(BEST_MODEL_DIR)[0]))

        incorrect_samples = []
        for data, target in test_loader:
            data = data.view(-1, 28, 28)
            if model.predict(data)[0] != target.item():
                incorrect_samples.append((data, target))
                if len(incorrect_samples) == num_samples:
                    break

        incorrect_samples = list(zip(*incorrect_samples))
        model.visualize(torch.stack(
            incorrect_samples[0]), torch.stack(incorrect_samples[1]))


if __name__ == '__main__':

    train_loader, test_loader = Data().generate()
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    trainer = Trainer(model, 'cpu', optimizer)

    train_losses = []
    test_losses = []
    best_acc = 0.0
    best_loss = Helper.get_best_loss()

    for epoch in range(EPOCHS):
        train_loss = trainer.train(train_loader, epoch + 1)
        test_loss, acc = trainer.test(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            Helper.delete_files_in_path(BEST_MODEL_DIR)
            Helper.save_model(model, rf'{BEST_MODEL_DIR}/epoch-{epoch+1}--loss-{test_loss:.3f}--accuracy-{acc*100}.pt')
            best_loss = test_loss

        if Helper.early_stop(test_losses):
            print(f'Early Stopping at epoch {epoch + 1}')
            break

    Helper.visualize_incorrect()

