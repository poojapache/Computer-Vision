"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    pts_u = np.mean(points[:,0])
    pts_v = np.mean(points[:,1])
    a = np.identity(3)
    b = np.identity(3)
    a[0, 0] = 1/np.std(points[:,0])
    a[1, 1] = 1/np.std(points[:,1])
    b[0,2] = -pts_u
    b[1,2] = -pts_v
    T = np.matmul(a,b)
    u_v = np.matmul(T, np.transpose(np.hstack((points, np.ones((points.shape[0],1))))))
    points_normalized = np.transpose(u_v)[:,0:2]
    # raise NotImplementedError(
    #     "`normalize_points` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    F_orig = np.matmul(np.transpose(T_b),np.matmul(F_norm,T_a))
    # raise NotImplementedError(
    #     "`unnormalize_F` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    size = points_a.shape[0]
    a_norm, ta = normalize_points(points_a)
    b_norm, tb = normalize_points(points_b)
    u = a_norm[:, 0:1]
    v = a_norm[:, 1:]
    u_p = b_norm[:, 0:1]
    v_p = b_norm[:, 1:]
    a = np.hstack((u*u_p, v*u_p, u_p, u*v_p, v*v_p, v_p, u, v))
    b = -np.ones(size)
    F_norm = np.reshape(np.append(np.linalg.lstsq(a, b)[0], [1]), newshape=(3, 3))
    c,d,vh = np.linalg.svd(F_norm)
    d[2] = 0
    F = unnormalize_F(np.matmul(c*d, vh), ta, tb)
    # raise NotImplementedError(
    #     "`estimate_fundamental_matrix` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
