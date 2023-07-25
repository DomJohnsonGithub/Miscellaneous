import numpy as np


class ChaosTest:
    def __init__(self, ncut):
        self.ncut = ncut

    def pq(self, data, c):
        imax = len(data)
        p = np.zeros(imax)
        q = np.zeros(imax)
        p[1] = data[1] * np.cos(c)
        q[1] = data[1] * np.sin(c)
        for i in range(2, imax):
            p[i] = p[i - 1] + data[i - 1] * np.cos(c * (i - 1))
            q[i] = q[i - 1] + data[i - 1] * np.sin(c * (i - 1))

        return p, q

    def Mn_c(self, data, c):
        p, q = self.pq(data, c)
        N = len(data) - self.ncut
        Mn = np.zeros(self.ncut)
        for n in range(1, self.ncut):
            Mn[n] = np.mean([(p[j + n] - p[j]) ** 2 + (q[j + n] - q[j]) ** 2 for j in range(1, N)])

        return Mn

    def Vosc_c(self, data, c):
        E = np.mean(data)
        return [E ** 2 * (1 - np.cos(n * c)) / (1 - np.cos(c)) for n in range(1, self.ncut + 1)]

    def Dn_c(self, data, c):
        return self.Mn_c(data, c) - self.Vosc_c(data, c)

    def correlation_method(self, data):
        # Main method that needs to be called
        epsilon = np.arange(1, self.ncut + 1)
        c_range = np.random.uniform(np.pi / 5, (0.8 * np.pi) + 1e-5, size=100)
        K_c = [np.corrcoef(epsilon, self.Dn_c(data, c)) for c in c_range]

        return np.median(K_c)