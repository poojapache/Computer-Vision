#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    length = len(X)
    half = feature_width//2
    fvs = []
    result = []
    for i in range(0, length):
        fv = (image_bw[(Y[i] - half + 1):(Y[i] - half + 1 + feature_width),(X[i] - half + 1):(X[i] - half + 1 + feature_width)])
        if(fv.size == feature_width * feature_width):
            fv = fv.astype(np.float32)
            fv = fv/(np.linalg.norm(fv))
            fv = fv.reshape(1, feature_width * feature_width)
            result.append(fv)
    fvs = np.vstack(result)

    # raise NotImplementedError('`compute_normalized_patch_descriptors` ' +
    #     'function in`part2_patch_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
