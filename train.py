"""
train
"""
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchmetrics

from models.alexnet import AlexNet
# from utils.load_dataset import CrackDataset


@dataclass
class Configs:
    """_summary_
    """
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.01


train_transform = transforms.Compose([
    # Resizing the image as the VGG only take 224 x 244 as input size
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),  # Flip the data horizontally
    # TODO if it is needed, add the random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

if __name__ == "__main__":
    configs = Configs()
    model = AlexNet()
    # dataset = CrackDataset()
    train_dataset = ImageFolder(
        root='datasets/crack', transform=train_transform)
    # valid_dataset = ImageFolder(root='valid')
    optimizer = Adam(model.parameters(), lr=configs.learning_rate)
    criterion = BCEWithLogitsLoss()

    train_loader = DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=True)
    for epoch in range(configs.epochs):
        loop = tqdm(train_loader)
        for bid, batch in enumerate(loop):
            x, y = batch[0], batch[1]
            # forward pass
            preds: Tensor = model(x)
            # print(y)
            # print(preds.reshape(-1))
            loss = criterion(preds.reshape(-1), y.float())

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = torchmetrics.functional.accuracy(
                preds.reshape(-1), y, task="binary", num_classes=2)

        # print(f'Epoch [{epoch + 1}/30], Loss: {loss.item():.4f}')
            loop.set_description(f"Epoch [{epoch}/{configs.epochs}]")
            loop.set_postfix(loss=loss.item(), acc=acc)
