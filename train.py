from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import model
from pileogram import PileogramDataset
import visualizer

NORMAL = "./data/normal"
REPEATS = "./data/repeats"
CHIMERIC = "./data/chimeric"

EPOCHS = 15
BATCH = 4
PARAM_PATH = 'models/params.pt'

types = {
    0: 'RP',
    1: 'CH',
    2: 'NM',
}


def print_confusion(conf_rep, conf_chim, conf_norm):
    print("%42s" % ('Predicted'))
    print(" " * 21 + "_" * 33)
    print(" " * 20 + "|%10s|%10s|%10s|" % ('Repeats', 'Chimeric', 'Normal'))
    print(" " * 9 + "|" + "%10s" % ('Repeats') + "|%10d|%10d|%10d|"
          % (conf_rep[0], conf_rep[1], conf_rep[2]))
    print("True" + " " * 5 + "|" + "%10s" % ('Chimeric') + "|%10d|%10d|%10d|"
          % (conf_chim[0], conf_chim[1], conf_chim[2]))
    print(" " * 9 + "|" + "%10s" % ('Normal') + "|%10d|%10d|%10d|"
          % (conf_norm[0], conf_norm[1], conf_norm[2]))


def main():
    # torch.manual_seed(0)
    mode = 'train'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    ds = PileogramDataset(REPEATS, CHIMERIC, NORMAL, transform=transform)
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
    train_time_start = time()

    if mode == 'train':
        for epoch in range(EPOCHS):
            total_loss = 0.0
            iteration = 0
            net.train()

            for data in tqdm(dl_train, desc=f"Epoch {epoch + 1}", leave=True, ncols=100):
                iteration += 1
                inputs = data['image'].to(device)
                labels = data['label'].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            tqdm.write(f"\nEpoch {epoch + 1} train loss: {total_loss / iteration}")
            history_train.append((epoch + 1, total_loss / iteration))

            total_loss = 0.0
            iteration = 0
            total = 0
            correct = 0
            net.eval()

            for data in dl_val:
                iteration += 1
                images = data['image'].to(device)
                labels = data['label'].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}: Val loss = {total_loss / iteration}, Accuracy = {accuracy}")
            history_val.append((epoch + 1, total_loss / iteration))
            history_acc.append((epoch + 1, accuracy))

            if epoch == 0 or history_acc[-1] > history_acc[-2]:
                torch.save(net.state_dict(), PARAM_PATH)

        train_time_end = time()
        print(f"Finished Training. Training time: {train_time_end - train_time_start} s.")
        visualizer.draw_training_curve(history_train, history_val)
        visualizer.draw_accuracy_curve(history_acc)

    correct = 0
    total = 0
    net.load_state_dict(torch.load(PARAM_PATH))
    net.eval()
    guess_repeat = []
    guess_chim = []
    guess_normal = []
    eval_time_start = time()

    with torch.no_grad(), open('wrong.txt', 'w') as f:
        for data in dl_test:
            images = data['image'].to(device)
            labels = data['label'].to(device)
            paths = data['path'][0]
            # print(paths)
            # print(type(paths))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if labels == 0:
                guess_repeat.append(predicted.item())
                if predicted.item() != 0:
                    output = paths[:-4] + types[int(labels)] + '_' + types[predicted.item()] + paths[-4:] + '\n'
                    f.write(output)
            elif labels == 1:
                guess_chim.append(predicted.item())
                if predicted.item() != 1:
                    output = paths[:-4] + types[int(labels)] + '_' + types[predicted.item()] + paths[-4:] + '\n'
                    f.write(output)
            else:
                guess_normal.append(predicted.item())
                if predicted.item() != 2:
                    output = paths[:-4] + types[int(labels)] + '_' + types[predicted.item()] + paths[-4:] + '\n'
                    f.write(output)

    eval_time_end = time()
    print(f"Accuracy of the network on the test set: {100 * correct / total}%.")
    print(f"Evalutaion time: {eval_time_end - eval_time_start} s.")

    conf_repeat = (sum([l == 0 for l in guess_repeat]), sum([l == 1 for l in guess_repeat]),
                   sum([l == 2 for l in guess_repeat]))
    conf_chim = (sum([l == 0 for l in guess_chim]), sum([l == 1 for l in guess_chim]),
                 sum([l == 2 for l in guess_chim]))
    conf_normal = (sum([l == 0 for l in guess_normal]), sum([l == 1 for l in guess_normal]),
                   sum([l == 2 for l in guess_normal]))

    print_confusion(conf_repeat, conf_chim, conf_normal)


if __name__ == '__main__':
    main()
