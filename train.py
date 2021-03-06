# from tqdm import tqdm
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import models
from pileogram import PileogramDataset
# import visualizer

REGULAR_TRAIN = "./manual/regular"
REPEATS_TRAIN = "./manual/repetitive"
CHIMERIC_TRAIN = "./manual/chimeric"
JUNK_TRAIN = "./manual/junk"

REGULAR_TEST = "./megan_test/regular"
REPEATS_TEST = "./megan_test/repetitive"
CHIMERIC_TEST = "./megan_test/chimeric"
JUNK_TEST = "./megan_test/junk"

EPOCHS = 30
BATCH = 128
PARAM_PATH = 'trained_mnodels/params_res18_man_ef.pt'

types = {
    0: 'RP',
    1: 'CH',
    2: 'RG',
    3: 'JK',
}


def print_confusion(conf_rep, conf_chim, conf_norm, conf_junk):
    print("%42s" % ('Predicted'))
    print(" " * 21 + "_" * 33)
    print(" " * 20 + "|%10s|%10s|%10s|%10s|" % ('Repeats', 'Chimeric', 'Normal', "Junk"))
    print(" " * 9 + "|" + "%10s" % ('Repeats') + "|%10d|%10d|%10d|%10d|"
          % (conf_rep[0], conf_rep[1], conf_rep[2], conf_rep[3]))
    print("True" + " " * 5 + "|" + "%10s" % ('Chimeric') + "|%10d|%10d|%10d|%10d|"
          % (conf_chim[0], conf_chim[1], conf_chim[2], conf_chim[3]))
    print(" " * 9 + "|" + "%10s" % ('Normal') + "|%10d|%10d|%10d|%10d|"
          % (conf_norm[0], conf_norm[1], conf_norm[2], conf_norm[3]))
    print(" " * 9 + "|" + "%10s" % ('Junk') + "|%10d|%10d|%10d|%10d|"
          % (conf_junk[0], conf_junk[1], conf_junk[2], conf_junk[3]))


def main():
    start_time = time()
    torch.manual_seed(7)
    #np.random.seed(0)
    mode = 'train'
    #############
    # mode = 'test'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    ds = PileogramDataset(REPEATS_TRAIN, CHIMERIC_TRAIN, REGULAR_TRAIN, JUNK_TRAIN, transform=transform)
    num_samples = len(ds)
    val_size = round(num_samples * 0.2)
    train_size = num_samples - val_size
    ds_train, ds_val = random_split(ds, [train_size, val_size])
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    ds_test = PileogramDataset(REPEATS_TEST, CHIMERIC_TEST, REGULAR_TEST, JUNK_TEST, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    net = models.ResNet18(num_classes=4)
    # if device.type == 'cuda' and torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use cuda if possible
    # device = torch.device('cpu')  # Force using cpu
    print(f"Using device: {device}")
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=3e-5, betas=(0.9, 0.999))
    optimizer = optim.RMSprop(net.parameters(), lr=3e-5)
    history_train = []
    history_val = []
    acc_train = []
    acc_valid = []

    if mode == 'train':
        for epoch in range(EPOCHS):
            total_loss = 0.0
            iteration = 0
            total = 0
            correct = 0
            net.train()

            for data in dl_train:
                iteration += 1
                inputs = data['image'].to(device, non_blocking=True)
                labels = data['label'].to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()
                total_loss += loss.item()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            # if i % 100 == 99:
            #    print("Epoch: %2d, Step: %5d -> Loss: %.5f" %
            #          (epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0
            accuracy = 100*correct/total
            print(f"Epoch {epoch + 1}:\tTrain loss = {total_loss / iteration}\tAccuracy = {round(accuracy, 2)}%")
            history_train.append((epoch + 1, total_loss / iteration))
            acc_train.append((epoch+1, accuracy))

            total_loss = 0.0
            iteration = 0
            total = 0
            correct = 0
            net.eval()

            with torch.no_grad():
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
            print(f"Epoch {epoch + 1}:\tVal loss = {total_loss / iteration},\tAccuracy = {round(accuracy, 2)}%")
            history_val.append((epoch + 1, total_loss / iteration))
            acc_valid.append((epoch + 1, accuracy))

            if epoch == 0 or acc_valid[-1] > max(acc_valid[:-1]):
                torch.save(net.state_dict(), PARAM_PATH)

        training_time = time()
        print(f"Finished Training. Training time: {training_time - start_time} s")
#        visualizer.draw_training_curve(history_train, history_val)
#        visualizer.draw_accuracy_curve(acc_train, acc_valid)

    correct = 0
    total = 0
    net.load_state_dict(torch.load(PARAM_PATH))
    net.eval()
    guess_repeat = []
    guess_chim = []
    guess_regular = []
    guess_junk = []
    eval_time_start = time()

    with torch.no_grad(), open('wrong.txt', 'w') as f:
        for data in dl_test:
            images = data['image'].to(device, non_blocking=True)
            labels = data['label'].to(device, non_blocking=True)
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
            elif labels == 2:
                guess_regular.append(predicted.item())
                if predicted.item() != 2:
                    output = paths[:-4] + types[int(labels)] + '_' + types[predicted.item()] + paths[-4:] + '\n'
                    f.write(output)
            else:
                guess_junk.append(predicted.item())
                if predicted.item() != 3:
                    output = paths[:-4] + types[int(labels)] + '_' + types[predicted.item()] + paths[-4:] + '\n'
                    f.write(output)

    eval_time_end = time()
    print(f"Accuracy of the network on the test set: {100 * correct / total}%.")
    print(f"Evalutaion time: {eval_time_end - eval_time_start} s.")

    conf_repeat = (sum([l == 0 for l in guess_repeat]), sum([l == 1 for l in guess_repeat]),
                   sum([l == 2 for l in guess_repeat]), sum([l == 3 for l in guess_repeat]))
    conf_chim = (sum([l == 0 for l in guess_chim]), sum([l == 1 for l in guess_chim]),
                 sum([l == 2 for l in guess_chim]), sum([l == 3 for l in guess_chim]))
    conf_regular = (sum([l == 0 for l in guess_regular]), sum([l == 1 for l in guess_regular]),
                   sum([l == 2 for l in guess_regular]), sum([l == 3 for l in guess_regular]))
    conf_junk = (sum([l == 0 for l in guess_junk]), sum([l == 1 for l in guess_junk]),
                   sum([l == 2 for l in guess_junk]), sum([l == 3 for l in guess_junk]))

    print_confusion(conf_repeat, conf_chim, conf_regular, conf_junk)


if __name__ == '__main__':
    main()
