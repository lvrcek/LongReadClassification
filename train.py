import os
import shutil
from tqdm import tqdm
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import model
from CoverageDataset import CoverageDataset
import visualizer

NORMAL = "./subsampled/normal"
REPEATS = "./subsampled/repeats"
CHIMERIC = "./subsampled/chimeric"
# NON_CHIMERIC = "./original_data/non-chimeric"

EPOCHS = 15
BATCH = 4
PARAM_PATH = 'models/params.pt'


def main():
    start_time = time()
    torch.manual_seed(0)
    np.random.seed(0)
    mode = 'eval'

    #    if not os.path.isdir(NON_CHIMERIC):
    #        os.mkdir(NON_CHIMERIC)
    #        for file in os.listdir(REGULAR):
    #            shutil.copy(os.path.join(REGULAR, file), os.path.join(NON_CHIMERIC, file))
    #        for file in os.listdir(REPEAT):
    #            shutil.copy(os.path.join(REPEAT, file), os.path.join(NON_CHIMERIC, file))

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    ds = CoverageDataset(REPEATS, CHIMERIC, NORMAL, transform=transform)
    num_samples = len(ds)
    val_size = test_size = round(num_samples * 0.2)
    train_size = num_samples - val_size - test_size
    ds_train, ds_val, ds_test = random_split(ds, [train_size, val_size, test_size])
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=2)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2)

    net = model.AlexNet(num_classes=3)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use cuda if possible
    device = torch.device('cpu')  # Force using cpu
    print(f"Using device: {device}")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4)
    history_train = []
    history_val = []
    history_acc = []

    if mode == 'train':
        for epoch in range(EPOCHS):
            running_loss = 0.0
            total_loss = 0.0
            iter = 0
            for data in tqdm(dl_train, desc=f"Epoch {epoch + 1}"):
                net.train()
                iter += 1
                inputs = data['image'].to(device)
                labels = data['label'].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_loss += loss.item()

            # if i % 100 == 99:
            #    print("Epoch: %2d, Step: %5d -> Loss: %.5f" %
            #          (epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0
            print(f"Epoch {epoch + 1} train loss: {total_loss / iter}")
            history_train.append((epoch + 1, total_loss / iter))

            total_loss = 0.0
            iter = 0
            total = 0
            correct = 0
            for data in dl_val:
                net.eval()
                iter += 1
                images = data['image'].to(device)
                labels = data['label'].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}: Val loss = {total_loss / iter}, Accuracy = {accuracy}")
            history_val.append((epoch + 1, total_loss / iter))
            history_acc.append((epoch + 1, accuracy))

            if epoch == 0 or history_acc[-1] > history_acc[-2]:
                torch.save(net.state_dict(), PARAM_PATH)

        training_time = time()
        print(f"Finished Training. Training time: {training_time - start_time} s")
        visualizer.draw_training_curve(history_train, history_val)
        visualizer.draw_accuracy_curve(history_acc)

    correct = 0
    total = 0
    net.load_state_dict(torch.load(PARAM_PATH))
    net.eval()
    guess_repeat = []
    guess_chim = []
    guess_normal = []
    with torch.no_grad():
        for data in dl_test:
            images = data['image'].to(device)
            labels = data['label'].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if labels == 0:
                guess_repeat.append(predicted.item())
            elif labels == 1:
                guess_chim.append(predicted.item())
            else:
                guess_normal.append(predicted.item())

    evaluation_time = time()
    print(f"Accuracy of the network on the test set: {100 * correct / total}%. ")
    #          f"Evalutaion time: {evaluation_time - training_time} s")
    heat_repeat = (sum([l == 0 for l in guess_repeat]), sum([l == 1 for l in guess_repeat]),
                   sum([l == 2 for l in guess_repeat]))
    heat_chim = (sum([l == 0 for l in guess_chim]), sum([l == 1 for l in guess_chim]),
                 sum([l == 2 for l in guess_chim]))
    heat_normal = (sum([l == 0 for l in guess_normal]), sum([l == 1 for l in guess_normal]),
                   sum([l == 2 for l in guess_normal]))

    print(heat_repeat)
    print(heat_chim)
    print(heat_normal)


if __name__ == '__main__':
    main()
