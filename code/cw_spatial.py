import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cw_functions import *

###--- Import spatial data ---###

### Read in the data
df = pd.read_csv('cw_data.csv')

### Generate the arrays needed from the dataframe
c = np.array(df["bicycle.theft"])
xi = np.array(df['xi'])
yi = np.array(df['yi'])
N = len(c)
coords = [(xi[i],yi[i]) for i in range(N)]

### Subsample the original data set
subsample_factor = 3
idx = subsample(N, subsample_factor, seed=42)
G = get_G(N,idx)
data = G @ c


###--- MCMC ---####

### Set MCMC parameters
l = 2
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|c)
log_target = log_poisson_target
log_likelihood = log_poisson_likelihood


# TODO: Complete Spatial Data questions (e), (f).


### Plotting examples
plot_2D(c, xi, yi, title='Bike Theft Data')                   # Plot bike theft count data
plot_2D(data, xi[idx], yi[idx], title='Subsampled Data')      # Plot subsampled data