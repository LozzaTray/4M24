import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import norm
import matplotlib.cm as cm
import copy
from tqdm import tqdm


def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2)/(2*pow(l, 2)))


def subsample(N, factor, seed=None):
    assert factor>=1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i,idx[i]] = 1
    return G


def probit(v):
    return np.array([-1 if x < 0 else 1 for x in v]) # was wrong


def predict_t(samples):
    """Returns probability t^* is equal to 1"""
    prob_t_one = norm.cdf(samples)
    return np.mean(prob_t_one, axis=0)


###--- Density fuctions ---###

def log_prior(u, C):
    """returns log-prior: log p(u)"""
    N = len(u)
    N_term = (N / 2) * np.log(2 * np.pi)

    _sign, log_det = np.linalg.slogdet(C)
    det_term = (log_det) / 2

    C_inv = np.linalg.inv(C + 1e-6 * np.eye(N))
    u_term = (u.T @ C_inv @ u) / 2

    return - (N_term + det_term + u_term)


def log_continuous_likelihood(u, v, G):
    """returns log-likelihood: log p(v|u)"""
    M = len(v)
    M_term = 0.5 * M * np.log(2 * np.pi)

    diff = v - G @ u
    v_term = (diff.T @ diff) / 2
    return - (M_term + v_term)


def log_probit_likelihood(u, t, G):
    """returns log-likelihood: log p(t|u)"""
    Gu = G @ u
    tGu = np.multiply(t, G @ u)
    vec = norm.logcdf(tGu)
    return np.sum(vec)


def log_poisson_likelihood(u, c, G):
    """returns likelihood p(counts|u)"""
    u_tilde = G @ u

    exp_term = np.sum(np.exp(u_tilde))
    cu_term = np.dot(c, u_tilde)

    return - exp_term + cu_term



def log_continuous_target(u, y, K, G):
    return log_prior(u, K) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K, G):
    return log_prior(u, K) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K, G):
    return log_prior(u, K) + log_poisson_likelihood(u, c, G)


###--- MCMC ---###

def grw(log_target, u0, data, K, G, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        data - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    print("Performing GRW...")

    X = []
    acc = 0
    u_prev = u0

    N = K.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    lt_prev = log_target(u_prev, data, K, G)

    for i in tqdm(range(n_iters)):

        z = np.random.randn(N)
        u_new = u_prev + beta * Kc @ z # Propose new sample - use prior covariance, scaled by beta

        lt_new = log_target(u_new, data, K, G)

        log_alpha = min(lt_new - lt_prev, 0) # Calculate acceptance probability based on lt_prev, lt_new
        log_uniform_draw = np.log(np.random.random())

        # Accept/Reject
        if log_uniform_draw <= log_alpha:# Compare log_alpha and log_uniform_draw to accept/reject sample
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


def pcn(log_likelihood, u0, y, K, G, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    print("Performing pCN...")

    X = []
    acc = 0
    u_prev = u0

    N = K.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    ll_prev = log_likelihood(u_prev, y, G)

    for i in tqdm(range(n_iters)):

        z = np.random.randn(N)
        scaling = np.sqrt(1 - pow(beta, 2))
        u_new = scaling * u_prev + beta * Kc @ z # Propose new sample using pCN proposal

        ll_new = log_likelihood(u_new, y, G)

        log_alpha = min(ll_new - ll_prev, 0) # Calculate pCN acceptance probability
        log_uniform_draw = np.log(np.random.random())

        # Accept/Reject
        if log_uniform_draw <= log_alpha: # Compare log_alpha and log_u to accept/reject sample
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


###--- Plotting ---###

def plot_3D(u, x, y, title=None):
    """Plot the latent variable field u given the list of x,y coordinates"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    if title:  plt.title(title)
    plt.show()


def plot_2D(counts, xi, yi, title=None, colors='viridis'):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.max(counts)])
    fig.colorbar(im)
    if title:  plt.title(title)
    plt.show()

def plot_2D_unlimited(counts, xi, yi, title=None, colors='viridis'):
    """Visualise count data given the index lists"""
    scaling = np.max(np.abs(counts))
    Z = - (scaling + 1) * np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[np.min(counts), np.max(counts)])
    fig.colorbar(im)
    if title:  plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_result(u, data, x, y, x_d, y_d, title=None):
    """Plot the latent variable field u with the observations,
        using the latent variable coordinate lists x,y and the
        data coordinate lists x_d, y_d"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax.scatter(x_d, y_d, data, marker='x', color='r')
    if title:  plt.title(title)
    plt.show()