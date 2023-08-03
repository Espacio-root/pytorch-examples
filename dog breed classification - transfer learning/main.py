import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import os
import requests
import tarfile
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

EPOCHS = 10
LEARNING_RATE = 1e-3
DATA_DIR = r'data'
BEST_MODEL_DIR = r'model'

plt.style.use('ggplot')

class Data:

    def __init__(self, batch_size=64, test_split=0.2):
        self.batch_size = batch_size
        self.test_split = test_split
        class_names = os.listdir(os.path.join(DATA_DIR, 'train'))
        self.class_names = [name.split('-')[1] for name in class_names]

    def download(self, dir):
        tar_path = os.path.join(os.getcwd(), dir, 'images.tar')
        if not os.path.exists(dir):
            os.makedirs(dir)

        '''Download the dataset'''
        if not os.path.exists(tar_path):

            url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
            res = requests.get(url, stream=True)

            with open(tar_path, 'wb') as fp:
                for chunk in res.iter_content(chunk_size=1024):
                    fp.write(chunk)

        '''Extract the dataset'''
        if not len(os.listdir(dir)) > 1:
            with tarfile.open(tar_path) as fp:
                fp.extractall(dir)

    def organize_dirs(self, dir):
        for category in os.listdir(dir):

            category_path = os.path.join(dir, category)
            split_idx = int(len(os.listdir(category_path)) * self.test_split)

            if len(os.listdir(category_path)) == 0:
                continue

            train_path = os.path.join(DATA_DIR, 'train', category)
            test_path = os.path.join(DATA_DIR, 'test', category)

            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(test_path):
                os.makedirs(test_path)

            for i, file in enumerate(os.listdir(category_path)):
                if i < split_idx:
                    shutil.move(os.path.join(category_path, file), test_path)
                else:
                    shutil.move(os.path.join(category_path, file), train_path)

    def generate(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_data = ImageFolder(os.path.join(
            DATA_DIR, 'train'), transform=train_transform)
        test_data = ImageFolder(os.path.join(
            DATA_DIR, 'test'), transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader


class Model(nn.Module):

    def __init__(self, num_classes=120):
        super(Model, self).__init__()
        self.model = self._initialize_resnet()
        classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 120 dog breeds
        )
        self.model.fc = classifier

    def _initialize_resnet(self):
        model = models.resnet18(pretrained=True)

        # freeze all the layers
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze the last layer
        for param in model.layer4.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        return self.model(x)


class Trainer:

    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optim = optimizer

    def train(self, train_loader, epoch):
        self.model.train()
        pbar = tqdm(train_loader)

        for data, target in pbar:

            output = self.model(data)
            self.optim.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optim.step()

            pbar.set_description(f'Epoch: {epoch} | Loss: {loss.item():.4f}')

        return loss.item()

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                total_loss += F.cross_entropy(output,
                                              target, reduction='sum').item()

                probabilities = F.softmax(output, dim=1)
                correct += torch.eq(probabilities.argmax(
                    dim=1).view_as(target), target).sum().item()

            average_loss = total_loss / len(test_loader.dataset)
            accuracy = correct / len(test_loader.dataset)

            return average_loss, accuracy


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
        if os.path.exists(BEST_MODEL_DIR) and len(os.listdir(BEST_MODEL_DIR)) > 0:
            return float(os.listdir(BEST_MODEL_DIR)[0].split('--loss-')[1].split('--')[0])
        else:
            return float('inf')

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
    def visualize(model, num_images=16, incorrect=False):
        model.eval()
        total = 0
        correct = 0
        
        if num_images % 4 != 0: raise ValueError('num_images must be divisible by 4')

        data = Data()
        _, test_loader = data.generate()
        class_names = data.class_names

        _, ax = plt.subplots(num_images // 4, 4, figsize=(20, 10))

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                probabilities = F.softmax(output, dim=1)

                for j in range(data.size()[0]):
                    predicted = class_names[probabilities[j].argmax(dim=0)]
                    actual = class_names[target[j]]
                    
                    if predicted == actual:
                        correct += 1
                        if incorrect:
                            continue

                    ax[total // 4, total % 4].imshow(data[j].permute(1, 2, 0).numpy())
                    ax[total // 4, total % 4].set_title(f'Predicted: {predicted}\nActual: {actual}', fontsize=10)
                    ax[total // 4, total % 4].axis('off')

                    total += 1
                    if total == num_images:
                        plt.suptitle(f'Accuracy: {correct}/{num_images} = {correct/num_images * 100:.2f}%')
                        plt.tight_layout()
                        plt.show()
                        return
                    
BEST_MODEL_DIR = os.path.join(os.getcwd(), BEST_MODEL_DIR)
if not os.path.exists(BEST_MODEL_DIR): os.mkdir(BEST_MODEL_DIR)


if __name__ == '__main__':
    data = Data()
    data.download(DATA_DIR)
    data.organize_dirs(os.path.join(DATA_DIR, 'Images'))
    train_loader, test_loader = data.generate()

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(model, optimizer)

    test_losses = []
    best_loss = Helper.get_best_loss()
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        trainer.train(train_loader, epoch+1)
        test_loss, accuracy = trainer.test(test_loader)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            Helper.delete_files_in_path(BEST_MODEL_DIR)
            Helper.save_model(
                model, rf'{BEST_MODEL_DIR}/epoch-{epoch+1}--loss-{test_loss:.3f}--accuracy-{accuracy*100}.pt')
            best_loss = test_loss

        if Helper.early_stop(test_losses):
            print(f'Early Stopping at epoch {epoch + 1}')
            break

        print(f'Epoch={epoch+1}\tLoss={test_loss:.2f}\tAccuracy={accuracy*100:.2f}%')
    
    # model = Model(120)
    # pth = os.listdir(BEST_MODEL_DIR)[0]
    # model = Helper.load_model(model, os.path.join(BEST_MODEL_DIR, pth))
    # Helper.visualize(model=model, num_images=16)
