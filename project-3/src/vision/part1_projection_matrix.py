import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    size,_ = points_2d.shape
    a = np.ones((size,1))
    b = np.zeros((size,4))
    arr1 = np.hstack((points_3d, a, b, points_2d[:, 0:1] * -1 * points_3d))
    arr2 = np.hstack((b, points_3d, a, points_2d[:, 1:] * -1 * points_3d))
    arr3 = np.empty((size*2, arr1.shape[1]), dtype = arr1.dtype)
    arr3[0::2] = arr1
    arr3[1::2] = arr2
    M = np.reshape(np.append(np.linalg.lstsq(arr3, points_2d.flatten())[0],[1]), newshape=(3, 4))
    # raise NotImplementedError(
    #     "`calculate_projection_matrix` function in "
    #     + "`projection_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    X = points_3d[:,0]
    Y = points_3d[:,1]
    Z = points_3d[:,2]
    deno = (P[2,0]*X + P[2,1]*Y + P[2,2]*Z + P[2,3])
    a = (P[0,0]*X + P[0,1]*Y + P[0,2]*Z + P[0,3])/deno
    b = (P[1,0]*X + P[1,1]*Y + P[1,2]*Z + P[1,3])/deno
    projected_points_2d = np.stack((a,b),axis = 1)
    # raise NotImplementedError(
    #     "`projection` function in " + "`projection_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    cc = np.transpose(np.matmul(-np.linalg.inv(M[:, 0:3]), M[:, 3:]))[0]
    # raise NotImplementedError(
    #     "`calculate_camera_center` function in "
    #     + "`projection_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
