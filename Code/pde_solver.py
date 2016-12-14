# A. Podkopaev, N. Puchkin, I. Silin

# Solver for numerical solution of PDE

import numpy as np

class PDE_Solver():
    def __init__(self, sigma=1, V=None, f=None, T=0, n=10, a_x=10, M=10):
        self.sigma = sigma      # float, must be positive; diffusion coefficient
        self.V = V      # function of 2 variables: x, t; dissipation
        self.f = f      # function of 1 variable: x; initial condition
        self.T = T      # float; final moment of time
        self.n = n      # int; number of time steps
        self.a_x = a_x      # float; right bound of coordinate segment
        self.M = M  # int; number of coordinate segments on the final iteration
        self.u = None       # numerical solution
        self.x = np.linspace(-1 * a_x, a_x, M + 1)
    
    # Finds a numerical solution of PDE
    def Solve(self):
        T = self.T
        n = self.n
        dt = 1. * T / n
        a_x = self.a_x
        M = self.M
        h_x = 2. * a_x / M
        sigma = self.sigma
        
        tau = np.linspace(0, T, n + 1)
        #w = self.Trapezoid_Weights(n, dt)
        w = self.Simpson_Weights(n, dt)
    
        x = np.linspace(-1 * (n + 1) * a_x, (n + 1) * a_x, (n + 1) * M + 1)
        F = self.f(x)
        
        #mu = self.Trapezoid_Weights(M, h_x)
        mu = self.Simpson_Weights(M, h_x)
    xi = np.linspace(-a_x, a_x, M + 1)
        alpha = 1. / 4 / sigma / dt
        p = np.sqrt(alpha / np.pi) * np.exp(-1. * alpha * xi**2)
        mu = mu * p
                    
        for k in range(n, 0, -1):
            V = self.V(x, tau[n - k])
            b = np.exp(-1. * w[k] * V * dt)
            Phi = F * b
            F = self.Convolve(Phi, mu)
            x = np.linspace(-1 * k * a_x, k * a_x, k * M + 1)
        
        V = self.V(x, T)
        self.u = F * np.exp(-1. * w[0] * V * dt)
    
        return
    
    # Weights corresponding to trapezoid rule quadrature
    def Trapezoid_Weights(self, number_of_segments, step):
        weights = step * np.ones(number_of_segments + 1)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        
        return weights
    
    # Convolution via FFT sum_j Phi[i, j] mu_j
    # Phi[i, j] = Phi[i + j]
    def Convolve(self, Phi, mu):
        c1 = Phi[:self.M]
        c2 = Phi[self.M:]
        c = np.concatenate([c2, c1], axis=0)
        len_c = c.shape[0]
        len_mu = mu.shape[0]
        b = np.concatenate([mu[::-1], np.zeros(len_c - len_mu)], axis=0)
        F = np.fft.ifft(np.fft.fft(c) * np.fft.fft(b)).real
        return F[:len_c - self.M]

    # Weights corresponding to Simpson rule quadrature
    # Number of segments must be even!!!
    def Simpson_Weights(self, number_of_segments, step):
        repeats = number_of_segments // 2
        weights = np.tile(np.array([2, 4]), repeats)
        weights = np.concatenate([weights, np.ones(1)], axis=0)
        weights[0] = 1
        weights *= (step / 3)
        
        return weights