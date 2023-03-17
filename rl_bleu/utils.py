# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import matplotlib.pyplot as plt


class score:
    def __init__(self, **kwargs):
        self.arr = []
        self.path = kwargs['path']
        self.xlabel = kwargs['xlabel']
        self.ylabel = kwargs['ylabel']

    def update(self, val):
        self.arr.append(val)
        self.save()

    def save(self):
        plt.plot(self.arr)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.savefig(self.path, dpi=300)
        plt.clf()
