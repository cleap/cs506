#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def approx_svd(U, S, Vh, N):
    """
    An approximation of A using the first N
    principal components
    """
    smat = np.diag(S[:N])
    return np.matmul(U[:,:N], np.dot(smat, Vh[:N,:]))

def approx_jl(A, m):
    """
    An approximation of A using m randomly generated
    components
    """
    d, n = A.shape
    P = np.random.normal(0, 1, (m, d)) / np.sqrt(m)
    comp = P @ A # m x n
    return P.T @ comp

def main():
    img = np.asarray(Image.open("image.jpg"))
    height, width = img.shape
    rank = min(height, width)
    U, S, Vh = np.linalg.svd(img, full_matrices=False)
    N_APPROX = 10
    N_COMPS = np.logspace(0, np.log2(rank), num=N_APPROX, base=2.0, dtype=int)

    fig, axs = plt.subplots(2, 5, figsize=(8.5,4), sharey=True, sharex=True)
    for i, n in enumerate(N_COMPS):
        row = i // 5
        col = i % 5
        B = approx_svd(U, S, Vh, n)
        axs[row, col].set_axis_off()
        axs[row, col].imshow(B, cmap="gray")
        axs[row, col].set_title(f"{n} components")
    fig.tight_layout()
    plt.suptitle("SVD Image Compression")
    plt.savefig("svd_result.pdf", bbox_inches="tight")

    fig, axs = plt.subplots(2, 5, figsize=(8.5,4), sharey=True, sharex=True)
    for i, n in enumerate(N_COMPS):
        row = i // 5
        col = i % 5
        B = approx_jl(img, n)
        axs[row, col].set_axis_off()
        axs[row, col].imshow(B, cmap="gray")
        axs[row, col].set_title(f"{n} components")
    fig.tight_layout()
    plt.suptitle("JL Image Compression")
    plt.savefig("jl_result.pdf", bbox_inches="tight")
    plt.show()
if __name__=="__main__":
    main()
