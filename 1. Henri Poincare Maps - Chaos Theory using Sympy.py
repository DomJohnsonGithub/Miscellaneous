import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from scipy.signal import argrelextrema, argrelmin, welch
import seaborn as sns
from datetime import datetime
import talib as ta
import warnings
from sympy import symbols, Eq
from sympy.solvers import solve
import yfinance as yf
from scipy.signal import savgol_filter
from gtda.time_series import SingleTakensEmbedding
from scipy.signal import periodogram
from tqdm import tqdm
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


def fetch_data(symbol, from_date, to_date, drop_cols, cols_to_drop):
    """ Fetch OHLC data."""
    df = yf.download(symbol, from_date, to_date)
    if drop_cols:
        df.drop(columns=cols_to_drop, inplace=True)

    return df


def outlier_treatment(data, lookback, n, method):
    """Use moving average to get a residual series from the
        original dataframe. We use the IQR and quantiles to
        make anomalous data-points nan values. Then we replace
        these nan values using interpolation with a linear method.
    """
    ma = pd.DataFrame(index=data.index)  # moving averages of each column
    for i, j in data.items():
        ma[f"{i}"] = ta.SMA(j.values, timeperiod=lookback)

    res = data - ma  # residual series

    Q1 = res.quantile(0.25)  # Quantile 1
    Q3 = res.quantile(0.75)  # Quantile 3
    IQR = Q3 - Q1  # IQR

    lw_bound = Q1 - (n * IQR)  # lower bound
    up_bound = Q3 + (n * IQR)  # upper bound

    res[res <= lw_bound] = np.nan  # set values outside range to NaN
    res[res >= up_bound] = np.nan

    res = res.interpolate(method=method)  # interpolation replaces NaN values

    prices = pd.DataFrame((res + ma))  # recompose original dataframe
    prices.dropna(inplace=True)  # drop NaN values

    return prices


def ami_optimal_time_delay(data, lags):
    dataframe = pd.DataFrame(data)
    for i in range(1, lags + 1):
        dataframe[f"Lag_{i}"] = dataframe.iloc[:, 0].shift(i)
    dataframe.dropna(inplace=True)

    bins = int(np.round(2 * (len(dataframe)) ** (1 / 3), 0))

    def calc_mi(x, y, bins):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    def mutual_information(dataframe, lags, bins):
        mutual_information = []
        for i in tqdm(range(1, lags + 1)):
            mutual_information.append(calc_mi(dataframe.iloc[:, 0], dataframe[f"Lag_{i}"], bins=bins))

        return np.array(mutual_information)

    average_mi = mutual_information(dataframe, lags, bins)
    first_minima = argrelmin(average_mi)[0][0]
    return average_mi, first_minima


if __name__ == "__main__":

    # Fetch OHLC Data
    symbol = "EURUSD=X"  # ticker
    from_date = datetime(2000, 1, 1)
    to_date = datetime.now()
    drop_columns = ["Adj Close", "Volume"]

    df = fetch_data(symbol=symbol, from_date=from_date,
                    to_date=to_date, drop_cols=True, cols_to_drop=drop_columns)

    # Treat Outliers
    df = outlier_treatment(df, lookback=10, n=2, method="linear")

    # ----------------------------------
    # Taken Embedded Time Series Matrix
    # average mutual information - time delay
    _, tau = ami_optimal_time_delay(df.Close, lags=100)

    te = SingleTakensEmbedding(parameters_type='search', n_jobs=11, dimension=3, time_delay=int(tau))
    taken_matrix = te.fit_transform(df.Close)

    print(taken_matrix)

    # ----------------------------------

    # DIAGONAL HYPERPLANE INTERSECTION
    x = np.array([1, 1.6])
    y = ([1, 1.6])
    xx, yy = np.meshgrid(x, y)
    z = np.array([[1, 1.3], [1.3, 1.6]])

    hyperplane = np.array([xx, yy, z])
    hyperplane2 = np.reshape(hyperplane.T, (4, 3))
    hyperplane2 = hyperplane2[:-1, :]  # we only need 3 points

    p0, p1, p2 = hyperplane2
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]  # first vector
    vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]  # second vector
    u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]  # cross product

    point1 = np.array(p1)
    normal1 = np.array(u_cross_v)
    d1 = -point1.dot(normal1)
    print('\nplane equation:\n{:1.4f}x + {:1.4f}y + {:1.4f}z + {:1.4f} = 0'.format(normal1[0], normal1[1], normal1[2],
                                                                                   d1))

    z1 = (-normal1[0] * xx - normal1[1] * yy - d1) * 1. / normal1[2]

    # SOLVING SYSTEMS OF PARAMETRIC EQUATIONS
    t = symbols("t")
    x, y, z = symbols("x, y, z")
    g = symbols("g")

    plane = Eq(normal1[0] * x + normal1[1] * y + normal1[2] * z + d1, g)  # create plane
    print(plane)

    # Points that pass from above to below the plane (not below to above because I want it to be transveral)
    above_or_below = np.array([solve(plane.subs([(x, taken_matrix[i][0]),
                                                 (y, taken_matrix[i][1]),
                                                 (z, taken_matrix[i][2])]), g) \
                               for i in tqdm(range(len(taken_matrix)))])

    above1, below1 = [], []
    for i in tqdm(range(1, len(above_or_below))):
        if above_or_below[i - 1] >= 0 and above_or_below[i] < 0:
            above1.append(taken_matrix[i - 1])
            below1.append(taken_matrix[i])
    above1 = np.array(above1)
    below1 = np.array(below1)

    # Parametric Equations for x_coords, y_coords and z_coords
    xs1 = [Eq((above1[i][0] - below1[i][0]) * t + below1[i][0] - x, 0) for i in range(len(above1))]
    ys1 = [Eq((above1[i][1] - below1[i][1]) * t + below1[i][1] - y, 0) for i in range(len(above1))]
    zs1 = [Eq((above1[i][2] - below1[i][2]) * t + below1[i][2] - z, 0) for i in range(len(above1))]

    # Solve Equations
    xs1 = np.array([solve(i, x) for i in xs1])
    ys1 = np.array([solve(i, y) for i in ys1])
    zs1 = np.array([solve(i, z) for i in zs1])
    ts1 = np.array([solve(plane.subs([(g, 0), (x, np.squeeze(i)),
                                      (y, np.squeeze(j)), (z, np.squeeze(k))]), t) \
                    for i, j, k in zip(xs1, ys1, zs1)])  # plug x, y and z eqn's into plane to get t

    # Get x, y and z co-ordinates
    xs1 = np.array([solve(Eq(np.squeeze(i), x).subs(t, np.squeeze(j)), x) for i, j in zip(xs1, ts1)])
    ys1 = np.array([solve(Eq(np.squeeze(i), y).subs(t, np.squeeze(j)), y) for i, j in zip(ys1, ts1)])
    zs1 = np.array([solve(Eq(np.squeeze(i), z).subs(t, np.squeeze(j)), z) for i, j in zip(zs1, ts1)])

    points_on_plane1 = np.concatenate([xs1, ys1, zs1], axis=1)  # put coordinates together

    # Intersecting points from above to below
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(xx, yy, z1, alpha=0.2, color="seagreen")
    ax.plot(taken_matrix[:, 0], taken_matrix[:, 1], taken_matrix[:, 2], c="black", lw=0.6, alpha=0.5)
    ax.scatter(points_on_plane1[:, 0], points_on_plane1[:, 1], points_on_plane1[:, 2], c="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    # Linearization of the Poincare Map for XY, XZ & YZ
    m1, b1 = np.polyfit(points_on_plane1[:, 0].astype(float), points_on_plane1[:, 1].astype(float), deg=1)
    m2, b2 = np.polyfit(points_on_plane1[:, 0].astype(float), points_on_plane1[:, 2].astype(float), deg=1)
    m3, b3 = np.polyfit(points_on_plane1[:, 1].astype(float), points_on_plane1[:, 2].astype(float), deg=1)

    # Poincare Map
    fig = plt.figure(dpi=50)
    plt.scatter(points_on_plane1[:, 0], points_on_plane1[:, 1], c="black", s=40, label="x, y")
    plt.scatter(points_on_plane1[:, 0], points_on_plane1[:, 2], c="red", s=40, label="x, z")
    plt.scatter(points_on_plane1[:, 1], points_on_plane1[:, 2], c="blue", s=40, label="y, z")
    plt.plot(points_on_plane1[:, 0], m1 * points_on_plane1[:, 0] + b1, c="black")  # line of best fit
    plt.plot(points_on_plane1[:, 0], m2 * points_on_plane1[:, 0] + b2, c="red")  # line of best fit
    plt.plot(points_on_plane1[:, 1], m3 * points_on_plane1[:, 1] + b3, c="blue")  # line of best fit
    plt.suptitle("POINCARE MAP")
    plt.legend(loc="best")
    plt.show()

    print("--------------------------------------")
    # Eigen Values from beginning and end points
    black_line = np.array([[np.min(points_on_plane1[:, 0]), m1 * np.min(points_on_plane1[:, 0]) + b1],
                           [np.max(points_on_plane1[:, 0]), m1 * np.max(points_on_plane1[:, 0]) + b1]])
    red_line = np.array([[np.min(points_on_plane1[:, 0]), m2 * np.min(points_on_plane1[:, 0]) + b2],
                         [np.max(points_on_plane1[:, 0]), m2 * np.max(points_on_plane1[:, 0]) + b2]])
    blue_line = np.array([[np.min(points_on_plane1[:, 1]), m3 * np.min(points_on_plane1[:, 1]) + b3],
                          [np.max(points_on_plane1[:, 1]), m3 * np.max(points_on_plane1[:, 1]) + b3]])
    eigvals1, eigvecs1 = np.linalg.eig(black_line.astype(float))
    eigvals2, eigvecs2 = np.linalg.eig(red_line.astype(float))
    eigvals3, eigvecs3 = np.linalg.eig(blue_line.astype(float))
    print("EIG VALS:")
    print(eigvals1)
    print(eigvals2)
    print(eigvals3)
    print("-------------")
    print("NORM EIG VALS:")
    print(np.linalg.norm(eigvals1))
    print(np.linalg.norm(eigvals2))
    print(np.linalg.norm(eigvals3))

    # PCA transform 2D Quiver Plot - does it tend to 0 (asymptotic stability)
    taken_matrix_pca = PCA(n_components=2).fit_transform(taken_matrix)
    fig = plt.figure(dpi=50)
    plt.quiver(taken_matrix_pca[:, 0], taken_matrix_pca[:, 1], np.gradient(taken_matrix_pca[:, 0]),
               np.gradient(taken_matrix_pca[:, 1]), color="black")
    plt.show()

    # Quiver Plot of 3D system
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.quiver(taken_matrix[:, 0], taken_matrix[:, 1], taken_matrix[:, 2], np.gradient(taken_matrix[:, 0]),
              np.gradient(taken_matrix[:, 1]), np.gradient(taken_matrix[:, 2]), color="black",
              length=1)
    plt.show()

    # -----------------------------------
    # FOR HYPERPLANE WHERE Z IS FLAT
    # x_min, x_max = np.min(taken_matrix[:, 0]), np.max(taken_matrix[:, 0])
    # y_min, y_max = np.min(taken_matrix[:, 1]), np.max(taken_matrix[:, 1])
    # z_min, z_max = np.min(taken_matrix[:, 2]), np.max(taken_matrix[:, 2])
    #
    # x = np.array([x_min, x_max])
    # y = np.array([y_min, y_max])
    # xx, yy = np.meshgrid(x, y)
    # n = 1.3
    # z = np.array([[n, n], [n, n]])
    #
    # hyperplane = np.array([xx, yy, z])
    # hyperplane = np.reshape(hyperplane.T, (4, 3))
    #
    # print("TAKEN EMBEDDED MATRIX")
    # print(taken_matrix)
    # print(taken_matrix.shape)
    #
    # hyperplane1 = hyperplane[:-1, :]
    # p0, p1, p2 = hyperplane1
    # x0, y0, z0 = p0
    # x1, y1, z1 = p1
    # x2, y2, z2 = p2
    #
    # ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]  # first vector
    # vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]  # second vector
    # u_cross_v = [uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx]  # cross product
    #
    # point1 = np.array(p1)
    # normal1 = np.array(u_cross_v)
    #
    # d1 = -point1.dot(normal1)
    # print('\nplane equation:\n{:1.4f}x + {:1.4f}y + {:1.4f}z + {:1.4f} = 0'.format(normal1[0], normal1[1], normal1[2], d1))
    #
    # xx, yy = np.meshgrid(x, y)
    # z1 = (-normal1[0] * xx - normal1[1] * yy - d1) * 1. / normal1[2]
    #
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.plot_surface(xx, yy, z1, alpha=0.2, color="seagreen")
    # ax.plot(taken_matrix[:, 0], taken_matrix[:, 1], taken_matrix[:, 2], c="black", lw=0.6, alpha=0.8)
    # ax.scatter(taken_matrix[:, 0], taken_matrix[:, 1], taken_matrix[:, 2], c="red", s=5)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()

    # # Get Points Above Hyperplane that Pass it to Below Coordinate and from Below to Above (remember transversality)
    # above1 = []; below1 = []  # above cross to below
    # for i in range(len(taken_matrix)):
    #     if taken_matrix[i][2] >= n and taken_matrix[i+1][2] < n:  # if current value above hyperplane and next one below
    #         above1.append(taken_matrix[i])
    #         below1.append(taken_matrix[i+1])
    # above1 = np.array(above1)
    # below1 = np.array(below1)
    #
    # below2 = []; above2 = [] # below cross to above
    # for i in range(1, len(taken_matrix)):
    #     if taken_matrix[i-1][2] <= n and taken_matrix[i][2] > n:  # if current value below hyperplane and next one above
    #         below2.append(taken_matrix[i-1])
    #         above2.append(taken_matrix[i])
    # below2 = np.array(below2)
    # above2 = np.array(above2)
    #
    # from sympy import symbols, Eq
    # from sympy.solvers import solve
    #
    # t = symbols("t")
    # x, y, z = symbols("x, y, z")
    # xs1 = [Eq((above1[i][0] - below1[i][0])*t + below1[i][0] - x, 0) for i in range(len(above1))]
    # ys1 = [Eq((above1[i][1] - below1[i][1])*t + below1[i][1] - y, 0) for i in range(len(above1))]
    # zs1 = [Eq((above1[i][2] - below1[i][2])*t + below1[i][2] - z, 0) for i in range(len(above1))]
    #
    # xs2 = [Eq((below2[i][0] - above2[i][0])*t + above2[i][0] - x, 0) for i in range(len(above1))]
    # ys2 = [Eq((below2[i][1] - above2[i][1])*t + above2[i][1] - y, 0) for i in range(len(above1))]
    # zs2 = [Eq((below2[i][2] - above2[i][2])*t + above2[i][2] - z, 0) for i in range(len(above1))]
    #
    # plane = Eq(normal1[0]*x + normal1[1]*y + normal1[2]*z + d1, 0)
    #
    # zs1 = np.array([solve(i, z) for i in zs1])
    # zs2 = np.array([solve(i, z) for i in zs2])
    # ts1 = [solve(plane.subs(z, np.squeeze(i)), t) for i in zs1]
    # ts2 = [solve(plane.subs(z, np.squeeze(i)), t) for i in zs2]
    #
    # xs1 = np.array([solve(i.subs(t, np.squeeze(j)), x)for i, j in zip(xs1, ts1)])
    # xs2 = np.array([solve(i.subs(t, np.squeeze(j)), x)for i, j in zip(xs2, ts2)])
    # ys1 = np.array([solve(i.subs(t, np.squeeze(j)), y)for i, j in zip(ys1, ts1)])
    # ys2 = np.array([solve(i.subs(t, np.squeeze(j)), y)for i, j in zip(ys2, ts2)])
    #
    # xs_ys1 = np.concatenate([xs1, ys1], axis=1)
    # xs_ys2 = np.concatenate([xs2, ys2], axis=1)
    # points_on_plane1 = np.concatenate([xs_ys1, np.full((len(xs_ys1), 1), n)], axis=1)
    # points_on_plane2 = np.concatenate([xs_ys2, np.full((len(xs_ys2), 1), n)], axis=1)
    #
    # print("---------")
    # print(points_on_plane1)
    # print("\n", points_on_plane2)
    #
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.plot_surface(xx, yy, z1, color="blue", alpha=0.3)
    # ax.plot(taken_matrix[:, 0], taken_matrix[:, 1], taken_matrix[:, 2], c="black", lw=0.6, alpha=0.4)
    # ax.scatter(points_on_plane1[:, 0], points_on_plane1[:, 1], points_on_plane1[:, 2], c="seagreen", label="Plane intersection above to below")
    # ax.scatter(points_on_plane2[:, 0], points_on_plane2[:, 1], points_on_plane2[:, 2], c="red", label="Plane intersection below to above")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.legend(loc="best")
    # plt.show()
    #
    # # POINCARE MAP
    # fig = plt.figure(dpi=50)
    # plt.scatter(points_on_plane1[:, 0], points_on_plane1[:, 1], c="black", label=f"Z={n}, Above to Below")
    # plt.scatter(points_on_plane2[:, 0], points_on_plane2[:, 1], c="red", label=f"Z={n} Below to Above")
    # plt.suptitle("POINCARE MAP")
    # plt.legend(loc="best")
    # plt.show()
