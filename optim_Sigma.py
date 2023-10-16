# This code minimizes the negative log-likelihood function with respect to
# initial variances (var_1, var_2, var_3) and the correlation coefficients
# (rho_12, rho_13, rho_23).


import numpy as np
import pandas as pd
from os.path import join, splitext
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sys import argv
import copy
import collections

def encode_marks(marks):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(marks)
    oh = np.zeros((len(marks), len(encoder.classes_)))
    oh[np.arange(len(marks)), encoded] = 1
    return oh, encoder.classes_

def sigmoid(mu_w, mu_l):
  b = np.log(10)/400 
  x = b*(mu_w-mu_l)
  return 1/(1+np.exp(-x))

def calculate_ratings(winners, losers, surface, prior_rate, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23, bound, d, b= np.log(10)/400, cov_renew = True):
    
    prior_ratings = defaultdict(lambda :(np.array([[prior_rate],[prior_rate],[prior_rate]],dtype = 'float64')))
    prior_var = defaultdict(lambda :(np.array([[variance_1],[variance_2],[variance_3]],dtype = 'float64')))
    
    
    total_discrepancy = 0
    for game in np.arange(winners.shape[0]):
        # current winners & losers
        cur_winner = winners[game]
        cur_loser = losers[game]
        # current surface [1,3]
        cur_surface = surface[game]
        # current player
        current_player = [cur_winner, cur_loser]
        # check whether the player play
        for player in current_player:
            if player not in prior_ratings:
                prior_ratings[player]
                prior_var[player]
        # save the prior rating
        new_rating = prior_ratings.copy()
        new_var = prior_var.copy()
        # winners & loser rating
        winner = new_rating[cur_winner]
        loser = new_rating[cur_loser]
        winner_var = new_var[cur_winner]
        loser_var = new_var[cur_loser]
        
        # calculate p(x)
        mu_w = cur_surface.dot(winner)[0] # float
        mu_l = cur_surface.dot(loser)[0] #float

        p_w = sigmoid(mu_w, mu_l) # float
        p_l = 1-p_w
 
        # calculate discrepancy
        discrepancy = -np.log(p_w)
        total_discrepancy += discrepancy
        #Calculate C
        C = 1/(1 + (b**2)*p_w*p_l*(cur_surface.dot(winner_var) + cur_surface.dot(loser_var)))
        #Calculate K
        rho_mat = np.array([[1, rho_12, rho_13],
                  [rho_12, 1, rho_23],
                  [rho_13, rho_23, 1]])        
        S_wi = np.sqrt(cur_surface.dot(winner_var))
        S_li = np.sqrt(cur_surface.dot(loser_var))
        cur_rho = cur_surface.dot(rho_mat)
        K_w = S_wi*np.sqrt(winner_var).T*cur_rho.T*b*C
        K_l = S_li*np.sqrt(loser_var).T*cur_rho.T*b*C
        # update the winner player & loser player rating
        new_rating[cur_winner] = new_rating[cur_winner] + K_w.T*(1-p_w)
        new_rating[cur_loser] = new_rating[cur_loser] + K_l.T*(-p_l)
        #Calculate new win prob and new C
        mu_w_new = cur_surface.dot(new_rating[cur_winner])[0] # float
        mu_l_new = cur_surface.dot(new_rating[cur_loser])[0] #float
        p_w_new = sigmoid(mu_w_new, mu_l_new)
        p_l_new = 1-p_w_new
        C_new = 1/(1 + (b**2)*p_w_new*p_l_new*(cur_surface.dot(winner_var) + cur_surface.dot(loser_var)))
        #Calculate L
        L = np.square(cur_rho)
        #winners
        L_w = (b**2)*p_w_new*p_l_new*L*winner_var.T*C_new
        L_w = L_w/d
        #losers
        L_l = (b**2)*p_w_new*p_l_new*L*loser_var.T*C_new
        L_l = L_l/d
        # update the players variance
        if cov_renew:
          tmp_w = winner_var*(1-L_w).T
          tmp_l = loser_var*(1-L_l).T
          new_var[cur_winner] = np.array([[max(np.array([bound**2]),tmp_w[0])[0]],[max(np.array([bound**2]),tmp_w[1])[0]],[max(np.array([bound**2]),tmp_w[2])[0]]],dtype = 'float64')
          new_var[cur_loser] = np.array([[max(np.array([bound**2]),tmp_l[0])[0]],[max(np.array([bound**2]),tmp_l[1])[0]],[max(np.array([bound**2]),tmp_l[2])[0]]],dtype = 'float64')
          prior_var = new_var
        # update prior rating
        prior_ratings = new_rating
    #return ratings, total_discrepancy
    return prior_ratings, prior_var, total_discrepancy/winners.shape[0]

train = pd.read_csv(argv[1])
winners = train['winner_name'] #Seperate the data into winners and losers
losers = train['loser_name']

train = train.reset_index()
surface = encode_marks(train['surface'])[0] #Encode the surfaces

def fun_GenElo(theta):
  variance_1, variance_2, variance_3, rho_12, rho_13, rho_23 = theta
  rating, prior_var, discrepancy = calculate_ratings(winners, losers, surface, 1500, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23, bound = 0, d = 1, b = np.log(10)/400, cov_renew = False)
  print(
        f'variance_1: {variance_1:.3f}; '
        f'variance_2: {variance_2:.3f}; '
        f'variance_3: {variance_3:.3f}; '
        f'rho_12: {rho_12:.3f}; '
        f'rho_13: {rho_13:.3f}; '
        f'rho_23: {rho_23:.3f}; '
        f'discrepancy: {discrepancy:.3f}')
  return discrepancy

def fun_vGenElo(theta, bound, d):
  variance_1, variance_2, variance_3, rho_12, rho_13, rho_23 = theta
  rating, prior_var, discrepancy = calculate_ratings(winners, losers, surface, 1500, variance_1, variance_2, variance_3, rho_12, rho_13, rho_23, bound=bound ,d=d, b = np.log(10)/400, cov_renew = True)
  print(
        f'variance_1: {variance_1:.3f}; '
        f'variance_2: {variance_2:.3f}; '
        f'variance_3: {variance_3:.3f}; '
        f'rho_12: {rho_12:.3f}; '
        f'rho_13: {rho_13:.3f}; '
        f'rho_23: {rho_23:.3f}; '
        f'discrepancy: {discrepancy:.3f}')
  return discrepancy

bound_list = [0, 80, 90]
d_list = [1, 2, 3, 4, 5]
file_name = 'optimize'
file_name +='.txt'
with open(file_name, 'w', encoding = "utf-8") as f:
  f.write('%2s, %2s, %2s, %2s, %2s, %2s, %2s, %2s\n'%('var_1', 'var_2', 'var_3', 'rho_12', 'rho_13', 'rho_23', 'fun_values', 'niter'))
  result = minimize(fun_GenElo,
                    np.array([120**2, 120**2, 120**2, 0.5, 0.5, 0.5]),
                    method = 'Nelder-Mead', tol = 0.01)
  print(f'success: {result.success:2d}, disrepancy: {result.fun:.4f}, niter: {result.nit:3d}')
  str_w = f'{result.x[0]:.4f}, {result.x[1]:.4f}, {result.x[2]:.4f}, {result.x[3]:.4f}, {result.x[4]:.4f}, {result.x[5]:.4f}, {result.fun:.4f}, {result.nit:4d}\\\\'
  f.write(str_w+"\n")
  f.write("\n")
  f.write("\n")
  f.write('%3s, %2s,%2s, %2s, %2s, %2s, %2s, %2s, %2s, %2s\n'%('bound','d','var_1', 'var_2', 'var_3', 'rho_12', 'rho_13', 'rho_23', 'fun_values', 'niter'))
  for i in range(0, len(bound_list)):
    for j in range(0, len(d_list)):
      result = minimize(fun_vGenElo,
                        np.array([120**2, 120**2, 120**2, 0.5, 0.5, 0.5]),
                        method = 'Nelder-Mead', args = (bound_list[i], d_list[j]), 
                        options={'maxiter':2000, 'xatol':0.02, 'fatol':0.5})
      print(f'success: {result.success:2d}, disrepancy: {result.fun:.4f}, niter: {result.nit:3d}')
      str_w = f'{bound_list[i]:3d}, {d_list[j]:2d}, {result.x[0]:.4f}, {result.x[1]:.4f}, {result.x[2]:.4f}, {result.x[3]:.4f}, {result.x[4]:.4f}, {result.x[5]:.4f}, {result.fun:.4f}, {result.nit:4d}\\\\'
      f.write(str_w+"\n")
