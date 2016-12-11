import numpy as np
from pde_solver import PDE_solver

class BlackScholes_Solver():
    def __init__(self):
        self.gamma = 1		# float, must be positive; volatility
        self.r = None		# float; risk-free interest rate
        self.g = None		# function of 1 variable: s; terminal condition
        self.T_ptime = 0    # terminal time

        self.a_x = 0        # right bound of coordinate segment
        
        self.c = None		# numerical solution; option price
    
	# Finds a numerical solution of Black-Scholes PDE
    def Solve(self):
        gamma = self.gamma
        r = self.r
        T_ptime = self.T_ptime

        a = self.a_x

        pde = PDE_solver()
        pde.solve()
        c = pde.u