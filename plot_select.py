import sys
from pathlib import Path as pPath
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, labels, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.labels = labels
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def main(plot_csv, selected_out=None):
    plot_in = pPath(plot_csv)
    out_path = plot_in.parent / f'{plot_in.stem}_selected.csv' if selected_out is None else selected_out

    raw = genfromtxt(plot_in, delimiter=',', skip_header=1)
    data = raw[:,0:2]
    clusters = raw[:,2]
    labels = [_[3] for _ in genfromtxt(plot_in, delimiter=',', names=True, dtype=None, encoding='utf-8')]

    #subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    #fig, ax = plt.subplots(subplot_kw=subplot_kw)
    fig, ax = plt.subplots(1, 1, figsize=(10,10), constrained_layout=True)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80, c=clusters, cmap='viridis')
    selector = SelectFromCollection(ax, pts, [f'_{x}' for x in range(1,21)])

    def accept(event):
        if event.key == "enter":
            # print("Selected points:")
            # print(selector.xys[selector.ind])
            # labels = [selector.labels[x] for x in selector.ind]
            # print(selector.ind)
            with out_path.open('w') as fh:
                fh.write('selected\n')
                fh.writelines([f'{labels[i]}\n' for i in selector.ind])
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()
            plt.close()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
