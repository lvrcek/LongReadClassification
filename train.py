import os
import shutil
from time import time
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import model
from CoverageDataset import CoverageDataset
from visualizer import draw_training_curve


REGULAR = "./original_data/regular"
REPEAT = "./original_data/repeat"
CHIMERIC = "./original_data/chimeric"
NON_CHIMERIC = "./original_data/non-chimeric"

EPOCHS = 5
BATCH = 8

def main():
    start_time = time()
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
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=2)
    dl_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=2)

    net = model.AlexNet()
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    history = []

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Use cuda if possible
    device = torch.device('cpu') # Force using cpu

    print(f"Using device: {device}")
    net.to(device)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(dl_train, 0):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            if i % 100 == 99:
                print("Epoch: %2d, Step: %5d -> Loss: %.5f" %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        history.append((epoch + 1, total_loss))

    training_time = time()
    print(f"Finished Training. Time for training: {training_time - start_time}")
    draw_training_curve(history)

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in dl_test:
            images = data['image']
            labels = data['label']
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    evaluation_time = time()
    print(f"Accuracy of the network on the test set: {100 * correct / total}. "
          f"Evalutaion time: {evaluation_time - training_time}")


if __name__ == '__main__':
    main()
