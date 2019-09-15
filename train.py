import torch
import model
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import shutil
import torch.optim as optim
import torch.nn as nn
from CoverageDataset import CoverageDataset

REGULAR = "./original_data/regular"
REPEAT = "./original_data/repeat"
CHIMERIC = "./original_data/chimeric"
NON_CHIMERIC = "./original_data/non-chimeric"


def main():
    if not os.path.isdir(NON_CHIMERIC):
        os.mkdir(NON_CHIMERIC)
        for file in os.listdir(REGULAR):
            shutil.copy(os.path.join(REGULAR, file), os.path.join(NON_CHIMERIC, file))
        for file in os.listdir(REPEAT):
            shutil.copy(os.path.join(REPEAT, file), os.path.join(NON_CHIMERIC, file))

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    ds = CoverageDataset(CHIMERIC, NON_CHIMERIC, transform=transform)
    num_samples = len(ds)
    train_size = round(num_samples * 0.75)
    test_size = num_samples - train_size
    ds_train, ds_test = random_split(ds, [train_size, test_size])
    dl_train = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0)

    net = model.AlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(dl_train, 0):
            inputs = data['image']
            labels = data['label']
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dl_test:
            images = data['image']
            labels = data['label']
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the test set: {100 * correct / total}")


if __name__ == '__main__':
    main()
