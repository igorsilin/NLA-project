# A. Podkopaev, N. Puchkin, I. Silin

# Solver for Black-Scholes PDE

import numpy as np
from pde_solver import PDE_Solver

class BlackScholes_Solver():
    def __init__(self, gamma=1, r=1, g=None, T_prime=0, t=0, n=10, exp_a_x=10, M=10):
        self.gamma = gamma		# float, must be positive; volatility
        self.r = r		        # float, must be positive; risk-free interest rate
        self.g = g		        # function of 1 variable: s; terminal condition
        self.T_prime = T_prime  # terminal time

        self.t = t  # time at which we want to know price

        self.n = n
        self.exp_a_x = exp_a_x        # should be > 1! right bound of coordinate segment;
        self.M = M

        self.s = None       # grid of stock prices
        self.c = None		# numerical solution; option price
    
	# Finds a numerical solution of Black-Scholes PDE
    def Solve(self):
        gamma = self.gamma
        r = self.r
        g = self.g
        T_prime = self.T_prime
        t = self.t

        f = lambda x : np.exp( 1. / (gamma**2) * (r - (0.5) * gamma ** 2) * x ) * g(np.exp(x))
        sigma = 0.5 * (gamma**2)
        V = lambda x, t : r + 0.5 / (gamma**2) * (r - (0.5) * gamma ** 2) ** 2

        pde = PDE_Solver( sigma=sigma, V=V, f=f, T=T_prime-t, n=self.n, a_x=np.log(self.exp_a_x), M=self.M )
        pde.Solve()
        self.s = np.exp(pde.x)
        self.c = np.exp( -1. / (gamma**2) * (r - (0.5) * gamma ** 2) * pde.x ) * pde.u

        return