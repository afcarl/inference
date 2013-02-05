from __future__ import division
import itertools
import sys
import random

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.misc as misc
import matplotlib.pyplot as plt

USAGE = 'Usage: %s <image file> <foreground file> <background file> <alpha=0.9>'

def learn_parameters(intensities):
    """
    Computes maximum likelihood estimates of mean and covariance for
    RGB image intensities.

    Input
    -----
    intensities : an N x 3 vector of RGB values

    Returns
    -------
    (mean, covariance)
    mean : a 3-element vector with the mean intensity in R/G/B
    covariance : a 3x3 matrix with the estimated covariance between R/G/B
    """
    mean = np.mean(intensities, 0)
    covariance = np.cov(intensities.T)
    return (mean, covariance)

def gibbs_segmentation(image, foreground, background, alpha,
                       burnin, collect_frequency, n_samples):
    """
    Uses Gibbs sampling to segment an image into foreground and background,
    using partial training data.

    Inputs
    ------
    image : a numpy array with the image. Should be Nx x Ny x 3
    foreground : a numpy array with a binary image. 1s indicate pixels that are
                 definitely foreground: this is only a partial segmentation!
    background : a numpy array with a binary image. 1s indicate pixels that are
                 definitely backgroun: this is only a partial segmentation!
    alpha : Ising model coupling parameter, which determines how strongly we
            encourage neighboring pixels to have the same label

    burnin : Number of iterations to run as 'burn-in' before collecting data
    collect_frequency : How many samples in between collected samples
    n_samples : how many samples to collect in total

    Returns
    -------
    A distribution of the collected samples: a numpy array with a value between
    0 and 1 (inclusive) at every pixel, where each value represents the
    probability that pixel belongs to the foreground.
    """
    (Nx, Ny, _) = image.shape
    # The distribution that you will return
    distribution = np.zeros((Nx, Ny))

    # Initialize binary estimates at every pixel randomly. Your code should
    # update this array pixel by pixel at each iteration.
    estimates = np.random.random((Nx, Ny)) > .5

    # Precompute useful quantities:
    phi = np.zeros((Nx, Ny, 2))
    obs_by_label = map(learn_parameters, [image[background], image[foreground]])

    for label in [0,1]:
        phi[:,:,label] = observation_model(image, obs_by_label[label])
    2/0

    base = 2*alpha - 1
    subtr = 1-alpha
    total_iterations = burnin + (collect_frequency * (n_samples - 1))
    pixel_indices = list(itertools.product(xrange(Nx),xrange(Ny)))
    labels = np.array([[0],[1]])

    for iteration in xrange(total_iterations):
        # Loop over entire grid, using a random order for faster convergence
        random.shuffle(pixel_indices)
        for (i,j) in pixel_indices:
            neighbor_values = get_neighbors(estimates, i, j)
            arrprobs = phi[i,j,:] * np.prod((neighbor_values==labels) * base + subtr,1)

            prob_foreground = arrprobs[1] / arrprobs.sum()
            estimates[i,j] = (np.random.random() < prob_foreground)

        if iteration >= burnin and (iteration-burnin) % collect_frequency == 0:
            distribution += estimates

    return distribution / n_samples


def get_neighbors(grid, x, y):
    """
    Returns values of grid at points neighboring (x,y)

    Inputs
    ------
    grid   : a 2d numpy array
    (x, y) : valid indices into grid

    Returns
    -------
    An array of values neighboring (x,y) in grid. Returns
    a 2-element array if (x,y) is at a corner,
    a 3-element array if (x,y) is at an edge, and
    a 4-element array if (x,y) is not touching an edge/corner
    """
    try:
        return grid[(x-1,x,x,x+1),(y,y-1,y+1,y)]
    except IndexError:
        out = []
        if y < grid.shape[1] - 1:
            out.append((x, y+1))
        if y > 0:
            out.append((x, y-1))
        if x < grid.shape[0] - 1:
            out.append((x+1, y))
        if x > 0:
            out.append((x-1, y))
        return grid[zip(*out)]

def observation_model(intensities, (mean, cov)):
    """
    Computes the probability of observing a set of RGB intensities for a
    given label, assuming a normal distribution for RGB with the given
    mean/covariance.

    Inputs
    ------
    intensities : an array whose last dimension corresponds to RGB intensities
    label : either 0 (foreground) or 1 (background)

    Returns
    -------
    The probability (density) of observing these intensities
    """
    centered = intensities - mean
    try:
        precision = np.linalg.inv(cov)
        k = cov.shape[0]
        return np.linalg.det(precision)*(2*np.pi)**(-k/2) * \
                np.exp(-inner1d(np.dot(centered, precision), centered))
    except np.linalg.LinAlgError:
        return np.exp(-inner1d(centered, centered) / cov[0,0]) / np.sqrt(2*np.pi * cov[0,0])

if __name__ == '__main__':
    if len(sys.argv) not in [4,5]:
        print(USAGE % sys.argv[0])
        sys.exit(1)

    files = sys.argv[1:4]
    images = map(misc.imread, files)
    (rgb_image, foreground, background) = images

    if len(sys.argv) == 4:
        alpha = 0.9
    else:
        alpha = float(sys.argv[4])

    mean_image = gibbs_segmentation(rgb_image, foreground != 0, background != 0, alpha, 5, 2, 100)

    plt.imshow(mean_image.astype('float64'), interpolation='nearest')
    plt.gray()
    plt.colorbar()
    print('Close displayed figure to terminate...')
    plt.show()
