import matplotlib.pyplot as plt
import numpy as np


def visualize_field(field, title="Field Intensity"):
    plt.imshow(np.abs(field)**2, cmap='inferno')
    plt.colorbar()
    plt.title(title)
    plt.show()