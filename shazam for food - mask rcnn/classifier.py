import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

import os
import shutil
from getimgs import GetImgs
from tqdm import tqdm
import matplotlib.pyplot as plt

from argparse import ArgumentParser

IMAGES_DIR = os.path.join(os.getcwd(), 'images')
DATA_DIR = os.path.join(os.getcwd(), 'data')
BEST_MODEL_DIR = os.path.join(os.getcwd(), 'model')

LEARNING_RATE = 1e-3
EPOCHS = 20

class Data:
    
    def __init__(self):
        pass
    
    def generate_dataset(self, classes=None, num_images_per_class=150, test_size=0.2):
        """
        Generates a dataset of images with a given number of classes and images per class.
        """
        if classes == None:
            with open('classes.txt', 'r') as fp:
                classes = fp.read().split('\n')
        self.classes = classes

        if not os.path.exists(IMAGES_DIR):
            GetImgs(classes, num_images_per_class, dir=IMAGES_DIR).run()
        
        for dir in os.listdir(IMAGES_DIR):
            class_dir = os.path.join(IMAGES_DIR, dir)
            test_dir = os.path.join(DATA_DIR, 'test', dir)
            train_dir = os.path.join(DATA_DIR, 'train', dir)
            total_num_files = len(os.listdir(class_dir))

            if not os.path.exists(test_dir): os.makedirs(test_dir)
            if not os.path.exists(train_dir): os.makedirs(train_dir)
            
            for i, file in enumerate(os.listdir(class_dir)):
                if i < total_num_files * test_size:
                    shutil.move(os.path.join(class_dir, file), os.path.join(test_dir, file))
                else:
                    shutil.move(os.path.join(class_dir, file), os.path.join(train_dir, file))
                    
        os.rmdir(IMAGES_DIR)
        self._verify_file_integrity()
        
    def _verify_file_integrity(self):
        """
        Verifies that the dataset is not corrupted.
        """
        # traverse through each file in data dir
        for parent_dir in os.listdir(DATA_DIR):
            for dir in os.listdir(os.path.join(DATA_DIR, parent_dir)):
                for file in os.listdir(os.path.join(DATA_DIR, parent_dir, dir)):
                    file_name = os.path.splitext(file)[0]
                    if file_name[0] == '.':
                        os.remove(os.path.join(DATA_DIR, parent_dir, dir, file))
                    
        
    @staticmethod
    def _get_number_of_classes():
        return len(os.listdir(os.path.join(DATA_DIR, 'test')))
        
    def get_train_loader(self):
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        train_data = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        
        return train_loader
    
    def get_test_loader(self):
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        test_data = ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
        
        return test_loader
    
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.model = self._initialize_resnet()
        num_classes = Data._get_number_of_classes()
        classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # 50 food items by default
        )
        self.model.fc = classifier

    def _initialize_resnet(self):
        model = resnet18(pretrained=True)

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
        test_loader = data.get_test_loader()
        class_names = data._get_number_of_classes()

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
    
    # set argparser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--incorrect', action='store_false')
    args = parser.parse_args()
    
    if args.train:
        data = Data()
        if not os.path.exists(DATA_DIR):
            data.generate_dataset()

        train_loader = data.get_train_loader()
        test_loader = data.get_test_loader()

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

    model = Model(Data()._get_number_of_classes())
    pth = os.listdir(BEST_MODEL_DIR)[0]
    model = Helper.load_model(model, os.path.join(BEST_MODEL_DIR, pth))
    Helper.visualize(model=model, num_images=16)
    
    
