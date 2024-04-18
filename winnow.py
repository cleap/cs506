#!/bin/python3
import math

import numpy as np
import matplotlib.pyplot as plt


def winnow(points, n, T, eta, rho, epsilon, alpha=0.0):
    # Assume M = rho in the notes...where rho is given as 1 (maximum 
    #  value of any entry in any feature vector)

    num_points = points.shape[0]

    r1 = 1
    r2 = 2 * rho

    # length of n dimensions
    l_vec = np.zeros((n,))
    u_vec = np.ones((n,))
    w_vec = np.ones((n,))

    t_total = 0

    for t in range(T):
        # find a misclassified point
        # <w^t, a_j> = <a_j, w^t> for two vectors w^t, a_j and a_j are the 
        # individual points
        classification_checks = points @ w_vec

        if (classification_checks >= alpha).all():
            t_total = t
            break
        else:
            # mask the classification checks (1 if less than 0; 0 otherwise)
            classification_checks_mask = classification_checks < alpha
            classification_checks_probability = \
                classification_checks_mask / np.sum(classification_checks_mask)

            # choose a random misclassified example to update the loss (gain)
            point_chosen_index = np.random.choice(num_points, 1,
                                                  p=classification_checks_probability)
            l_vec[:] = -points[point_chosen_index, :]

            ###########################################
            # the chosen expert (with w_vec) is done INDEPENDENTLY of the 
            # chosen feature above
            U = np.sum(u_vec)
            u_normalized = u_vec / U

            # choose a random index given w_vec as the probability distribution
            w_chosen_index = np.random.choice(n, 1, p=u_normalized)

            # then update the chosen expert
            w_vec[w_chosen_index] = r1 * u_vec[w_chosen_index] / U
            ########################################

            # need element-wise multiplication
            u_vec = np.multiply(u_vec, 1 - eta * (l_vec + r2 / 2) / r2)

    print(f"total steps t: {t_total}")
    print(f"w = {w_vec}")
    classification_checks = points @ w_vec
    print(f"classification_checks = {classification_checks}")


def winnow_lm(points, n, T, eta, rho, epsilon):
    """
    winnow large margin
    """

    winnow(points, n, T, eta, rho, epsilon, alpha=(epsilon / 2))


def main():
    points = np.array([
        [-1 / 2, 1 / 3, 1],
        [3 / 4, -1 / 4, 1],
        [1 / 2, -1 / 3, -1],
        [-3 / 4, 1 / 4, -1]
    ])

    # points are n + 1 dimensional (last dimension for class)
    # different points are stored row-wise
    n = points.shape[1] - 1

    # parameters
    rho = 1
    epsilon = 0.05
    eta = epsilon / (2 * rho)

    # print parameters 
    print(f"n = {n}")
    print(f"rho = {rho}")
    print(f"epsilon = {epsilon}")
    print(f"eta = {eta}")

    # FIXME: T bound from the survey and NOT from the lecture notes...
    T_numerator = 4 * np.power(rho, 2) * np.log(n)
    T_denominator = np.power(epsilon, 2)
    T = np.ceil(T_numerator / T_denominator)
    T = int(T)
    print(f"T = {T}")

    # multiply each point by its class
    # new_points = points[:, :n] * points[:, n]
    classes = points[:, n]
    new_points = np.multiply(points[:, :n], classes[:, np.newaxis])

    print()
    print("Normal Winnow:")
    winnow(new_points, n, T, eta, rho, epsilon)

    print()
    print(f"Winnow with margin epsilon/2={epsilon / 2}:")
    winnow_lm(new_points, n, T, eta, rho, epsilon)

    fig, ax = plt.subplots()
    xs = points[:, 0]
    ys = points[:, 1]
    marks = points[:, 2]
    
    none_color = "none"
    black_color = "black"
    
    points_dict = dict()
    points_dict[none_color] = {}
    points_dict[black_color] = {}
    points_dict[none_color]["points"] = {}
    points_dict[black_color]["points"] = {}
    points_dict[none_color]["points"]["x"] = \
        [x for (x, mark) in zip(xs, marks) if mark < 0]
    points_dict[none_color]["points"]["y"] = \
        [y for (y, mark) in zip(ys, marks) if mark < 0]
    points_dict[black_color]["points"]["x"] = \
        [x for (x, mark) in zip(xs, marks) if mark > 0]
    points_dict[black_color]["points"]["y"] = \
        [y for (y, mark) in zip(ys, marks) if mark > 0]
    points_dict[none_color]["label"] = \
        "class=-1"
    points_dict[black_color]["label"] = \
        "class=1"
    
    # colors = ["none" if label < 0 else "black"
    #           for label in points[:, 2]]
    # labels = ["class=-1" if label < 0 else "class=1"
    #           for label in points[:, 2]]
    # 
    # none_colors = [color for color in colors if color == "none"]
    # black_colors = [color for color in colors if color == "black"]
    # 
    # none_labels = [label for label in labels if label == "class=-1"]
    # black_labels = [label for label in labels if label == "class=1"]
    # 
    # ax.scatter(xs, ys, s=80, facecolors=colors, edgecolors="black",
    #                      label=labels)

    for color in [none_color, black_color]:
        ax.scatter(points_dict[color]["points"]["x"],
                   points_dict[color]["points"]["y"], 
                   s=80, 
                   facecolors=color, 
                   edgecolors="black",
                   label=points_dict[color]["label"])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot of Points")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
