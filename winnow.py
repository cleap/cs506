#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def winnow(points, n, T, eta):
    pass

def winnow_lm(poitns, n, T, eta):
    """
    winnow large margin
    """
    pass

def main():
    points = np.array([
        [-1/2,  1/3,  1],
        [ 3/4, -1/4,  1],
        [ 1/2, -1/3, -1],
        [-3/4,  1/4, -1]
    ])

    fig, ax = plt.subplots()
    xs = points[:,0]
    ys = points[:,1]
    colors = ["none" if label < 0 else "black" for label in points[:,2]]
    ax.scatter(xs, ys, s=80, facecolors=colors, edgecolors="black")

    plt.show()

if __name__=="__main__":
    main()
