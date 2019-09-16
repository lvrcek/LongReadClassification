from matplotlib import pyplot as plt

def draw_training_curve(history):
    plt.figure()
    epochs = [h[0] for h in history]
    loss = [h[1] for h in history]
    plt.plot(epochs, loss)
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()