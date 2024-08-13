"""A widget for showing images in TSNE clusters. """
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import TSNE
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Resize


if __name__ == '__main__':
    # generate some stub TSNE inputs
    dataset = MNIST(
        root='C:/Users/chiwe/Data/',
        transform=Resize((128, 128))
    )
    tsne_list = []  # inputs for TSNE
    image_list = []
    label_list = []
    for i_samp in range(300):
        img, label = dataset[i_samp]
        image_list.append(img)
        label_list.append(label)
        fake_tsne = torch.randn(10)
        fake_tsne[label] = fake_tsne[label] + 4
        tsne_list.append(fake_tsne)
    tsne_list = torch.stack(tsne_list).numpy()

    # build widget
    proj = TSNE().fit_transform(tsne_list)
    root = tk.Tk()
    fig, (left, right) = plt.subplots(figsize=(12, 6), ncols=2)
    left.scatter(
        *proj.T,
        s=30,
        c=label_list,
        marker='.',
        cmap='tab10',
        picker=5  # enables click callbacks
    )
    left.grid()
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

    # setup callback
    canvas.mpl_connect(
        'pick_event',
        lambda event: update_right(event.ind[0])
    )

    def update_right(idx: int) -> None:
        """Show the idx-th image. """
        right.imshow(image_list[idx], cmap='gray')
        canvas.draw()

    # run app
    root.mainloop()
