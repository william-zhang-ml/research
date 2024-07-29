"""Widget that shows a random CIFAR10 training batch. """
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid


PATH_TO_DATA = 'C:/Users/chiwe/Data/'


if __name__ == '__main__':
    root = tk.Tk()
    fig, axes = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, root)
    canvas_widget = canvas.get_tk_widget()
    data = CIFAR10(
        root=PATH_TO_DATA,
        train=True,
        transform=ToTensor()
    )
    loader = DataLoader(data, batch_size=64, shuffle=True)

    def refresh() -> None:
        """Plot a new batch in figure. """
        imgs, _ = next(iter(loader))
        grid = make_grid(imgs)
        axes.imshow(ToPILImage()(grid))
        canvas.draw()

    button = ttk.Button(
        root,
        text='Refresh',
        command=refresh
    )

    canvas_widget.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
    button.pack(padx=2, pady=2, fill=tk.X)
    refresh()

    root.mainloop()
