#!/usr/bin/python3

import numpy as np
import torch

from torch import nn
from typing import Tuple


SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use convolution with Sobel filters to compute the image gradient at each
    pixel.

    Args:
        image_bw: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    Ix = my_conv2d_numpy(image_bw, SOBEL_X_KERNEL).astype(np.float32)
    Iy = my_conv2d_numpy(image_bw, SOBEL_Y_KERNEL).astype(np.float32)
    # raise NotImplementedError('`compute_image_gradients` function in ' +
        # '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return Ix, Iy

def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.

    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    img             = image.copy()
    rows            = img.shape[0]
    columns         = img.shape[1]
    ksizeRows       = filter.shape[0]
    ksizeColumns    = filter.shape[1]
    imgTemp         = np.zeros([rows, columns], dtype = int)
    imgTemp         = img
    padding         = ((ksizeRows//2,ksizeRows//2),(ksizeColumns//2,ksizeColumns//2))
    img             = np.pad(img, pad_width = padding,mode='constant')

    for i in range(rows):
        for j in range(columns):
            imgTemp[i][j]     = np.sum(img[i:i+ksizeRows,j:j+ksizeColumns]*filter)
            
    filtered_image  = imgTemp

    # raise NotImplementedError(
    #     "`my_conv2d_numpy` function in `part1.py` needs to be implemented"
    # )
    
    ### END OF STUDENT CODE ####
    ############################

    return filtered_image

def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel.

    Args:
        ksize: dimension of square kernel
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel
    """

    norm_mu = int(ksize / 2)
    idxs_1d = torch.arange(ksize).float()
    exponents = -((idxs_1d - norm_mu) ** 2) / (2 * (sigma ** 2))
    gauss_1d = torch.exp(exponents)

    # make normalized column vector
    gauss_1d = gauss_1d.reshape(-1, 1) / gauss_1d.sum()
    gauss_2d = gauss_1d @ gauss_1d.T
    kernel = gauss_2d

    return kernel


def second_moments(
        image_bw: np.ndarray,
        ksize: int = 7,
        sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, then mixed derivatives,
    then compute the second moments (sx2, sxsy, sy2) at each pixel, using
    convolution with a Gaussian filter.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter

    Returns:
        sx2: array of shape (M,N) containing the second moment in the x direction
        sy2: array of shape (M,N) containing the second moment in the y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the y direction
    """

    Ix, Iy = compute_image_gradients(image_bw)

    Ix = torch.from_numpy(Ix)
    Iy = torch.from_numpy(Iy)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # combine along a new dimension
    channel_products = torch.stack((Ix2, Iy2, Ixy), 0).unsqueeze(0)

    # create second moments S_xx, S_yy and S_xy from I_xx, I_xy, I_yy
    Gk = get_gaussian_kernel_2D_pytorch(ksize=ksize, sigma=sigma)

    pad_size = ksize // 2
    conv2d_gauss = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=ksize,
        bias=False,
        padding=(pad_size, pad_size),
        padding_mode='zeros',
        groups=3
    )

    conv2d_gauss.weight = nn.Parameter(
        Gk.expand((3, 1, ksize, ksize))
    )
    second_moments = conv2d_gauss(channel_products)

    # compute corner responses
    sx2 = second_moments[:, 0, :, :].squeeze()
    sy2 = second_moments[:, 1, :, :].squeeze()
    sxsy = second_moments[:, 2, :, :].squeeze()

    sx2 = sx2.detach().numpy()
    sy2 = sy2.detach().numpy()
    sxsy = sxsy.detach().numpy()

    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * in equation S_xx = Gk * I_xx is a convolutional operation over a
    Gaussian kernel of size (k, k).
    You may call the second_moments function above to get S_xx S_xy S_yy in M.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
            ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score

    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
    detA = sx2 * sy2 - sxsy ** 2
    traceA = sx2 + sy2
    R = detA - alpha * (traceA)**2


    # raise NotImplementedError('`compute_harris_response_map` function in ' +
    #  '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return R


def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator.
    Please read this implementation, which will help you understand
    what’s happening in nms_maxpool_pytorch.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d
            score/response map
    """

    (m, n) = R.shape
    padded_image = np.pad(R, (((ksize - 1) // 2,), ((ksize - 1) // 2,)), 'constant', constant_values=(0,))
    maxpooled_R = np.zeros(R.shape)
    for h in range(m):
        for l in range(n):
            maxpooled_R[h, l] = np.max(padded_image[h: h + ksize, l: l + ksize])

    return maxpooled_R


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get top k interest points that are local maxima over (ksize,ksize)
    neighborhood.

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.

    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator

    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    median = np.median(R)
    R1 = R
    R1[R1 < median] = 0
    rows = R1.shape[0]
    cols = R1.shape[1]
    R1 = R1.reshape(1, 1, rows, cols)
    R1 = torch.Tensor(R1)
    pooling = nn.MaxPool2d(ksize, stride = 1 , padding = ((ksize - 1)//2,(ksize - 1)//2))
    M = pooling(R1)
    Mbinary = []
    R1 = R1.numpy()
    M = M.numpy()
    Mbinarized = np.where(R1 == M, 1, 0)
    R2 = R1 * Mbinarized
    R2 = R2.reshape(rows,cols)
    R2 = torch.Tensor(R2)
    coord = torch.nonzero(R2)
    coord = coord.numpy()
    xtemp = coord[:,1]
    ytemp = coord[:,0]
    conftemp = []
    R2 = R2.numpy()
    for i in range(0,len(coord)):
        conftemp.append(R2[coord[i][0]][coord[i][1]])

    sort_indexes = torch.argsort(torch.Tensor(conftemp), descending = True)[0:k]
    x = []
    y = []
    confidences = []
    for i in range(0,k):
        x.append(xtemp[sort_indexes[i]])
        y.append(ytemp[sort_indexes[i]])
        confidences.append(conftemp[sort_indexes[i]])

    x = np.array(x)
    y= np.array(y)
    confidences = np.clip(np.array(confidences),0,1)

    # raise NotImplementedError('`compute_harris_response_map` function in ' +
    #     '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, confidences


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,) representing x coord of interest points
        y: array of shape (k,) representing y coord of interest points
        c: array of shape (k,) representing confidences of interest points

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    img_h, img_w = img.shape[0], img.shape[1]
    x_valid = (x >= 7) & (x <= img_w - 9)
    y_valid = (y >= 7) & (y <= img_h - 9)
    valid_idxs = x_valid & y_valid
    x,y,c = x[valid_idxs], y[valid_idxs], c[valid_idxs]

    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the Harris Corner detector. You will find compute_harris_response_map(), 
    nms_maxpool_pytorch(), and remove_border_vals() useful. 
    Make sure to normalize your response map to fall within the range [0,1].
    The kernel size here is 7x7.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: maximum number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        c: array of dim (p,) containing the strength(confidence) of each
            interest point where p <= k.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    new_img = compute_harris_response_map(image_bw)
    x,y,c = nms_maxpool_pytorch(new_img,k,7)
    x,y,c = remove_border_vals(new_img,x,y,c)
    # raise NotImplementedError('`get_harris_interest_points` function in ' +
    #     '`part1_harris_corner.py` needs to be implemented')

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    
    return x, y, c