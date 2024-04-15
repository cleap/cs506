#!/bin/python3
import math

import numpy as np
import matplotlib.pyplot as plt

def winnow(points, n, T, eta, rho, epsilon):
    # FIXME: assume M = rho in the notes...where rho is given as 1  (maximum 
    #  value of any entry in any feature vector)
    
    num_points = points.shape[0]
    
    r1 = 1
    r2 = 2 * rho
    
    # FIXME: N+1 experts in MWVector(R1, R2) but N experts in Winnow...
    l_vec = np.zeros((n, ))
    u_vec = np.ones((n, ))
    w_vec = np.ones((n, ))
    
    print(f"l_vec: {l_vec}")
    print(f"u_vec: {u_vec}")
    
    for t in range(T):
        # find a misclassified point
        # <w^t, a_j> = <a_j, w^t> for two vectors w^t, a_j and a_j are the 
        # individual points
        classification_checks = points @ w_vec
        
        print(f"t: {t}")
        print(f"classification_checks: {classification_checks}")
        print((classification_checks >= 0.0).all())
        
        if (classification_checks >= 0.0).all():
            break
        else:
            # mask the classification checks (1 if less than 0; 0 otherwise)
            classification_checks_mask = classification_checks < 0
            classification_checks_probability = \
                classification_checks_mask / np.sum(classification_checks_mask)

            # choose a random misclassified example to update the loss (gain)
            point_chosen_index = np.random.choice(num_points, 1, 
                                                  p=classification_checks_probability)
            l_vec[:] = -points[point_chosen_index, :]

            ###########################################
            # FIXME: chosen expert is the same above that is misclassified?
            # FIXME: can't because w is length of n (length of point vector 
            #  without class...)
            print(f"u_vec: {u_vec}")
            
            U = np.sum(u_vec)
            u_normalized = u_vec / U
            
            print(u_normalized)

            # choose a random index given w_vec as the probability distribution
            w_chosen_index = np.random.choice(n, 1, p=u_normalized)

            # then update the chosen expert
            w_vec[w_chosen_index] = r1 * u_vec[w_chosen_index] / U

            # # then update the chosen expert
            # w_vec[point_chosen_index] = r1 * u_vec[point_chosen_index] / U
            ########################################
            
            print(f"END l_vec: {l_vec}")
        
            # need element-wise multiplication
            u_vec = np.multiply(u_vec, 1 - eta * (l_vec + r2 / 2) / r2)
            print(f"END u_vec: {u_vec}")
            
    print()
    print(f"w = {w_vec}")

def winnow_lm(points, n, T, eta, rho, epsilon):
    """
    winnow large margin
    """
    
    # FIXME: assume M = rho in the notes...where rho is given as 1  (maximum 
    #  value of any entry in any feature vector)

    num_points = points.shape[0]
    
    # ADDED CHANGE
    margin_bound = epsilon / 2

    r1 = 1
    r2 = 2 * rho

    # FIXME: N+1 experts in MWVector(R1, R2) but N experts in Winnow...
    l_vec = np.zeros((n, ))
    u_vec = np.ones((n, ))
    w_vec = np.ones((n, ))

    print(f"l_vec: {l_vec}")
    print(f"u_vec: {u_vec}")

    for t in range(T):
        # find a misclassified point
        # <w^t, a_j> = <a_j, w^t> for two vectors w^t, a_j and a_j are the 
        # individual points
        classification_checks = points @ w_vec

        print(f"t: {t}")
        print(f"classification_checks: {classification_checks}")
        print((classification_checks >= margin_bound).all())

        if (classification_checks >= margin_bound).all():
            break
        else:
            # mask the classification checks (1 if less than 0; 0 otherwise)
            classification_checks_mask = classification_checks < margin_bound
            classification_checks_probability = \
                classification_checks_mask / np.sum(classification_checks_mask)

            # choose a random misclassified example to update the loss (gain)
            point_chosen_index = np.random.choice(num_points, 1,
                                                  p=classification_checks_probability)
            l_vec[:] = -points[point_chosen_index, :]

            ###########################################
            # FIXME: chosen expert is the same above that is misclassified?
            # FIXME: can't because w is length of n (length of point vector 
            #  without class...)
            print(f"u_vec: {u_vec}")

            U = np.sum(u_vec)
            u_normalized = u_vec / U

            print(u_normalized)

            # choose a random index given w_vec as the probability distribution
            w_chosen_index = np.random.choice(n, 1, p=u_normalized)

            # then update the chosen expert
            w_vec[w_chosen_index] = r1 * u_vec[w_chosen_index] / U

            # # then update the chosen expert
            # w_vec[point_chosen_index] = r1 * u_vec[point_chosen_index] / U
            ########################################

            print(f"END l_vec: {l_vec}")

            # need element-wise multiplication
            u_vec = np.multiply(u_vec, 1 - eta * (l_vec + r2 / 2) / r2)
            print(f"END u_vec: {u_vec}")

    print()
    print(f"w = {w_vec}")


def main():
    points = np.array([
        [-1/2,  1/3,  1],
        [ 3/4, -1/4,  1],
        [ 1/2, -1/3, -1],
        [-3/4,  1/4, -1]
    ])
    
    epsilon = 0.05

    # points are n + 1 dimensional (last dimension for class)
    # different points are stored row-wise
    n = points.shape[1] - 1
    m = np.max(points)
    rho = 1
    # rho = m
    eta = epsilon / (2 * rho)

    # T_numerator = 4 * m * np.sqrt(np.log(n))
    # T_denominator = epsilon
    # T = np.ceil(np.power(T_numerator / T_denominator, 2.0/3.0))
    T_numerator = 4 * np.power(rho, 2) * np.log(n)
    T_denominator = np.power(epsilon, 2)
    T = np.ceil(T_numerator / T_denominator)
    T = int(T)
    print(f"T: {T}")
    
    # multiply each point by its class
    # new_points = points[:, :n] * points[:, n]
    classes = points[:, n]
    new_points = np.multiply(points[:, :n], classes[:, np.newaxis])

    # winnow(new_points, n, T, eta, rho, epsilon)
    winnow_lm(new_points, n, T, eta, rho, epsilon)
    
    # print(np.zeros((1, )) > 0)
    # print(np.array([1, 0, 0]).any() > 0)
    # print(np.ones((1, )) > 0)

    # fig, ax = plt.subplots()
    # xs = points[:,0]
    # ys = points[:,1]
    # colors = ["none" if label < 0 else "black" for label in points[:,2]]
    # ax.scatter(xs, ys, s=80, facecolors=colors, edgecolors="black")
    # 
    # plt.show()

if __name__=="__main__":
    main()
