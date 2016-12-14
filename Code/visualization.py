# A. Podkopaev, N. Puchkin, I. Silin

import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def g_eu_call(s, E):
    return np.maximum(s - E,0)

def g_eu_put(s, E):
    return np.maximum(E - s,0)

def g_cash_or_nothing_call(s,E,B):
    res = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if s[i] >= E:
            res[i] = B
    return res

def g_cash_or_nothing_put(s,E,B):
    res = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if s[i] < E:
            res[i] = B
    return res

def g_asset_call(s,E):
    res = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if s[i] >= E:
            res[i] = s[i]
    return res

def g_asset_put(s,E):
    res = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if s[i] < E:
            res[i] = s[i]
    return res

def d_plus(cur_t, s,E,r,gamma,T_prime):
    return (math.log(s / E) + (r + (gamma ** 2 / 2.)) * (T_prime - cur_t)) / (gamma * math.sqrt(T_prime - cur_t))

def d_minus(cur_t, s,E,r,gamma,T_prime):
    return (math.log(s / E) + (r - (gamma ** 2 / 2.)) * (T_prime - cur_t)) / (gamma * math.sqrt(T_prime - cur_t))

def c_eu_call(s, cur_t,E,r,gamma,T_prime,B):
    return s * norm.cdf(d_plus(cur_t, s,E,r,gamma,T_prime)) - E * math.exp(-r * (T_prime - cur_t)) * norm.cdf(d_minus(cur_t, s,E,r,gamma,T_prime))

def c_eu_put(s, cur_t,E,r,gamma,T_prime,B):
    return E * math.exp(-r * (T_prime - cur_t)) * norm.cdf(-d_minus(cur_t, s,E,r,gamma,T_prime)) - s * norm.cdf(-d_plus(cur_t, s,E,r,gamma,T_prime))

def c_cash_or_nothing_call(s, cur_t,E,r,gamma,T_prime,B):
    return B * np.exp(-r * (T_prime - cur_t)) * norm.cdf(d_minus(cur_t, s,E,r,gamma,T_prime))

def c_cash_or_nothing_put(s, cur_t,E,r,gamma,T_prime,B):
    return B * np.exp(-r * (T_prime - cur_t)) * norm.cdf(-d_minus(cur_t, s,E,r,gamma,T_prime))

def c_asset_or_nothing_call(s, cur_t,E,r,gamma,T_prime,B):
    return s * norm.cdf(d_plus(cur_t, s,E,r,gamma,T_prime))

def c_asset_or_nothing_put(s, cur_t,E,r,gamma,T_prime,B):
    return s * norm.cdf(-d_plus(cur_t, s,E,r,gamma,T_prime))

def get_time_grid(t_var, M, T_prime):
    tt = np.array(t_var)
    tt = np.reshape(tt, [1, tt.shape[0]])
    tt_row = tt
    for i in range(M):
        tt = np.append(tt, tt_row, axis = 0)
    tt = tt * T_prime 
    return tt

def visualize_analytical_solution(cur_type, costs, tt,E,r,gamma,T_prime,B, costs_grid,g,M):
    if cur_type == 'eu_call':
        cur_fun = c_eu_call
    if cur_type == 'eu_put':
        cur_fun = c_eu_put
    if cur_type == 'cash_or_nothing_call':
        cur_fun = c_cash_or_nothing_call
    if cur_type == 'asset_or_nothing_call':
        cur_fun = c_asset_or_nothing_call
    if cur_type == 'cash_or_nothing_put':
        cur_fun = c_cash_or_nothing_put
    if cur_type == 'asset_or_nothing_put':
        cur_fun = c_asset_or_nothing_put
    f_1 = np.zeros([costs.shape[0],costs.shape[1]-1])
    for j in range(costs.shape[1]-1):
        for i in range(costs.shape[0]):
            f_1[i, j] = cur_fun(costs[i, j + 1], tt[i, j],E,r,gamma,T_prime,B)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Analytical solution')
    surf = ax.plot_wireframe(tt, costs[:,1:],  f_1, color='purple')
    ###########
    if g == g_eu_call:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    if g == g_eu_put:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    if g == g_cash_or_nothing_call:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E,B),linewidth = 4,color='red')
    if g == g_cash_or_nothing_put:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E,B),linewidth = 4,color='red')
    if g == g_asset_call:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    if g == g_asset_put:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    ##########
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Option Value')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    
def visualize_numerical_solution(costs, tt, values, costs_grid, g,E,B,M,T_prime):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Numerical solution')
    surf = ax.plot_wireframe(tt, costs[:,1:], values[:,1:],color='purple')
    if g == g_eu_call:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    if g == g_eu_put:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    if g == g_cash_or_nothing_call:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E,B),linewidth = 4,color='red')
    if g == g_cash_or_nothing_put:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E,B),linewidth = 4,color='red')
    if g == g_asset_call:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    if g == g_asset_put:
        ax.plot(np.ones(M+1) * T_prime, costs_grid, g(costs_grid, E),linewidth = 4,color='red')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Option Value')
    plt.show()
    
def visualize_difference(cur_type, costs, tt, values,E,r,gamma,T_prime,B):
    if cur_type == 'eu_call':
        cur_fun = c_eu_call
    if cur_type == 'eu_put':
        cur_fun = c_eu_put
    if cur_type == 'cash_or_nothing_call':
        cur_fun = c_cash_or_nothing_call
    if cur_type == 'asset_or_nothing_call':
        cur_fun = c_asset_or_nothing_call
    if cur_type == 'cash_or_nothing_put':
        cur_fun = c_cash_or_nothing_put
    if cur_type == 'asset_or_nothing_put':
        cur_fun = c_asset_or_nothing_put
    f_1 = np.zeros([costs.shape[0],costs.shape[1]-1])
    for j in range(costs.shape[1]-1):
        for i in range(costs.shape[0]):
            f_1[i, j] = cur_fun(costs[i, j + 1], tt[i, j],E,r,gamma,T_prime,B)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Difference between numerical and analytical solutions')
    surf = ax.plot_wireframe(tt, costs[:,1:], values[:,1:] - f_1, color='purple')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Option Values Difference')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()



