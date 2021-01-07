import numpy as np
import matplotlib.pyplot as plt
from cw_functions import *


###--- Data Generation ---###
seed = 0
np.random.seed(seed)

### Inference grid defining {ui}i=1,Nx*Ny
Nx = 16 # equivalent of D
Ny = 16
N = Nx * Ny     # Total number of coordinates
points = [(x, y) for y in np.arange(Nx) for x in np.arange(Ny)]                # Indexes for the inference grid
coords = [(x, y) for y in np.linspace(0,1,Ny) for x in np.linspace(0,1,Nx)]    # Coordinates for the inference grid
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])    # Get x, y index lists
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists

### Data grid defining {vi}i=1,Dx*Dy subsampled from inference grid
subsample_factor = 4
idx = subsample(N, subsample_factor)
M = len(idx)                                                                   # Total number of data points

### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
l = 0.3
C = GaussianKernel(coords, l)
z = np.random.randn(N)
Cc = np.linalg.cholesky(C + 1e-6 * np.eye(N))
u = Cc @ z

### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)


###--- MCMC ---####

### Set MCMC parameters
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|v)
log_target = log_continuous_target
log_likelihood = log_continuous_likelihood

### Sample from prior for MCMC initialisation


# TODO: Complete Simulation questions (a), (b).


### Plotting examples
plot_3D(u, x, y, title="GP prior samples of u (l={})".format(l))  # Plot original u surface
plot_result(u, v, x, y, x[idx], y[idx], title="Simulated data v overlaid onto u surface (l={})".format(l))               # Plot original u with data v


###--- Probit transform ---###
t = probit(v)       # Probit transform of data


# TODO: Complete Simulation questions (c), (d).


### Plotting examples
plot_2D(probit(u), xi, yi, title='Original Data')     # Plot true class assignments
plot_2D(t, xi[idx], yi[idx], title='Probit Data')     # Plot data