# A. Podkopaev, N. Puchkin, I. Silin

# Solver for numerical solution of PDE

import numpy as np

class PDE_Solver():
    def __init__(self):
        self.sigma = 1		# float, must be positive
        self.V = None		# function of 2 variables: t, x; potential
        self.f = None		# function of 1 variable: x; initial condition
        self.T = 0			# final moment of time
        self.delta_t = 0	# time step
        self.n = 0			# number of time steps
        self.a_x = 0		# right bound of coordinate segment
        self.M = 0			# number of coordinate segments on the final iteration
        self.h_x = 0		# coordinate step
        self.u = None		# numerical solution
    
	# Finds a numerical solution of PDE
    def Solve(self):
        T = self.T
        dt = self.delta_t
        n = self.n
        a = self.a_x
        M = self.M
        h = self.h_x
        sigma = self.sigma
        
        tau = np.linspace(0, T, n + 1)
        w = self.Trapezoid_Weights(n, dt)
        
        x = np.linspace(-1 * (n + 1) * a, (n + 1) * a, (n + 1) * M + 1)
        F = self.f(x)
        
        mu = self.Trapezoid_Weights(M, h)
        xi = np.linspace(-a, a, M + 1)
        alpha = 1. / 4 / sigma / dt
        p = np.sqrt(alpha / np.pi) * np.exp(-1. * alpha * xi**2)
        mu = mu * p
                    
        for k in range(n, 0, -1):
            V = self.V(x, tau[n - k])
            b = np.exp(-1. * w[k] * V * dt)
            Phi = F * b
            F = self.Convolve(Phi, mu)
            x = np.linspace(-1 * k * a, k * a, (k + 1) * M + 1)
        
        V = self.V(x, T)
        self.u = F * np.exp(-1. * w[0] * V * dt)
    
        return
    
	# Weights corresponding to trapezoid rule quadrature
    def Trapezoid_Weights(self, number_of_segments, step):
        weights = step * np.ones(number_of_segments + 1)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        
        return weights
    
	# Convolution via FFT
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