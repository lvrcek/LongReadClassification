from matplotlib import pyplot as plt


def draw_training_curve(history_train, history_val):
    plt.figure()
    epochs = [h[0] for h in history_train]
    loss_train = [h[1] for h in history_train]
    loss_test = [h[1] for h in history_val]
    plt.plot(epochs, loss_train, label='train')
    plt.plot(epochs, loss_test, label='validation')
    plt.title("Model loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def draw_accuracy_curve(history_acc):
    plt.figure()
    epochs = [h[0] for h in history_acc]
    acc = [h[1] for h in history_acc]
    plt.plot(epochs, acc)
    plt.title("Model accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
