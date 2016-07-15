import matplotlib.pyplot as plt

def _plot_hist(self, x, label):
    """
    Generates a line graph of a 1-d input vector
    """

    plt.plot(x)
    plt.ylabel(label)
    plt.show()
