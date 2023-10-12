"""
## Reference



"""
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchmetrics import functional

from utils.argparse import parse
from models.alexnet import AlexNet
from models.lenet import LeNet5
from models.shallow import ShallowCNN


class CrackClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = F.binary_cross_entropy_with_logits
        self.acc = lambda y_hat, y: functional.accuracy(
            y_hat, y, task="binary")
        self.precision = lambda y_hat, y: functional.precision(
            y_hat, y, task="binary")
        self.recall = lambda y_hat, y: functional.recall(
            y_hat, y, task="binary")
        self.f1_score = lambda y_hat, y: functional.f1_score(
            y_hat, y, task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.model(x)
        y_hat = y_hat.reshape(-1)
        loss = self.loss(y_hat, y)
        acc = self.acc(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_precision", precision)
        self.log("train_recall", recall)
        self.log("train_f1_score", f1_score)
        return loss

    def evaluate(self, batch, mode):
        x, y = batch
        y = y.float()
        y_hat = self.model(x)
        y_hat = y_hat.reshape(-1)

        loss = self.loss(y_hat, y)
        acc = self.acc(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        self.log(f"{mode}_loss", loss, sync_dist=True)
        self.log(f"{mode}_acc", acc, sync_dist=True)
        self.log(f"{mode}_precision", precision, sync_dist=True)
        self.log(f"{mode}_recall", recall, sync_dist=True)
        self.log(f"{mode}_f1_score", f1_score, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.learning_rate)
        return optimizer


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(
        root='datasets/crack', transform=transform)
    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset, [0.8, 0.1, 0.1])
    train_loader = data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size, num_workers=4)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    arg_model, arg_log = parse()
    model_name = arg_model
    if arg_log == "local":
        logger = True
    elif arg_log == "wandb":
        logger = WandbLogger(log_model="all", name=model_name)
    if model_name == "lenet":
        model = LeNet5()
    elif model_name == "alexnet":
        model = AlexNet()
    else:
        model = ShallowCNN()

    classifier = CrackClassifier(model, learning_rate=1e-4)

    train_loader, val_loader, test_loader = load_data(batch_size=64)

    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger,
        devices=3,
        accelerator="gpu",
        profiler="simple",
        default_root_dir="logs"
    )
    trainer.fit(
        model=classifier, train_dataloaders=train_loader,
        val_dataloaders=val_loader)

    trainer.test(model=classifier, dataloaders=test_loader)
