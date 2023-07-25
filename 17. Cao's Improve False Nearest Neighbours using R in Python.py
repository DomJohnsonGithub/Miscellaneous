import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

nonlinearTseries = importr("nonlinearTseries")
data = numpy2ri.numpy2rpy(savgol_price)  # <------- any array of your choice

cao_emb_dim = nonlinearTseries.estimateEmbeddingDim(
    data,  # time series
    len(data),  # number of points to use, use entire series
    62,  # time delay  - calculated via average mutual information or auto-correlation function
    20,  # max no. of dimension
    0.95,  # threshold value
    0.1,  # max relative change
    True,  # do the plot
    "Computing the embedding dimension",  # main
    "dimension (d)",  # x_label
    "E1(d) & E2(d)",  # y_label
    ro.NULL,  # x_lim
    ro.NULL,  # y_lim
    1e-5  # add a small amount of noise to the original series to avoid the
          # appearance of false neighbours due to discretization errors.
          # This also prevents the method to fail with periodic signals, 0 for no noise
)

embedding_dimension = int(cao_emb_dim[0])