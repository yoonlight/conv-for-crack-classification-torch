from torchvision import transforms, datasets
from torch.utils import data
import split_dataset

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root='datasets/crack', transform=transform)
    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset, [0.8, 0.1, 0.1])
    train_loader = data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size, num_workers=4)

    return train_loader, val_loader, test_loader, test_dataset


if __name__ == "__main__":
    dataset = datasets.ImageFolder(
        root='datasets/crack')
    # print(dataset.imgs)
    dataset
    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset, [0.8, 0.1, 0.1])
    print(test_dataset.dataset.imgs)