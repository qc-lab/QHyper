import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

name = "karate_solvers_overview"

methods = ["dqm", "cqm", "gurobi", "louvain"]
folder = "demo/demo_output"
imgs = [f"{folder}/karate_{method}.png" for method in methods] + [
    f"{folder}/karate_adv_{i}.png" for i in range(4)
]
filename = f"{folder}/{name}.pdf"


def display_save_imgs(image_paths: list[str]) -> None:
    im_len = len(image_paths)
    sqrt = int(np.sqrt([im_len]))
    sqrt = sqrt if im_len / sqrt == sqrt else sqrt + 1
    # f_size = 6.0 if sqrt <= 4 else 10.0
    f_size = 35.0

    plt.axis("off")
    fig = plt.figure(figsize=(f_size, f_size))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(sqrt, sqrt),
        axes_pad=0.1,
    )
    imgs = [Image.open(path) for path in image_paths]

    for ax, im in zip(grid, imgs):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])

    p = PdfPages(filename)
    fig.savefig(p, format="pdf")
    p.close()


display_save_imgs(imgs)
