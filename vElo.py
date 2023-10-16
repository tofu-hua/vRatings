# This code performs the variance-incorporated Elo system described in 
# Algorithm 1 of the paper. It computes the accuracy rates based on train/test data 
# and the average negative loglikelihood value based on train data, 
# based on numerous initial sigma values. 
# The results of the update rule with constant variance are also reported.
# 
# The negative loglikelihood is sometimes called "discrepancy" in the code.


import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import copy
import os
from sys import argv
from locale import atof, atoi


"""# config"""
kwarg = {'rho_12':1,
         'rho_13':1,
         'rho_23':1,
         'start_mean': 1500,
         'update_var':False}

# Set the starting sigma 
init_sig = [50, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 140, 150, 160, 170, 175, 180, 200, 205, 210]

"""### encode Function"""
def encode_marks(marks):

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(marks)
    oh = np.zeros((len(marks), len(encoder.classes_)))

    oh[np.arange(len(marks)), encoded] = 1

    return oh, encoder.classes_



"""# input arguments
argv1: train data filename 
argv2: test data filename
argv3: lower bound on standard deviation, e.g. argv3 = 0, 70, 80; (argv3 corresponds to the B value in the paper)
argv4: additional factor controling variance reduction, e.g. argv4 = 1, 2, 5; (1/argv4 corresponds to the A value in the paper)
"""

"""### Data input"""

train = pd.read_csv(argv[1])
test = pd.read_csv(argv[2])
winners, losers = train['winner_name'], train['loser_name']
winners_val, losers_val = test['winner_name'], test['loser_name']

surface = encode_marks(train['surface'])[0] #Encode the surfaces
surface_val = encode_marks(test['surface'])[0]
par3 = argv[3]
par3_i = atoi(par3)
par4 = argv[4]
par4_i = atoi(par4)

folder =  'OUTPUT_vElo_b'+par3+'_d'+par4
folderS = folder+'/'

print(folder)
print(folderS)

try:
  os.mkdir(folderS)
except:
  pass

os.chdir(folder)


"""### more Functions"""

def sigmoid(mu_w, mu_l):
  b = np.log(10)/400 
  x = b*(mu_w-mu_l)
  return 1/(1+np.exp(-x))

def calculate_ratings(winners, losers, surface, var_1, var_2, var_3, rho_12, rho_13, rho_23, start_mean, file_name = None, update_var = True, mode = 'Train', prior_mean = None, prior_var = None):
  if file_name==None: file_name = mode + ('_both.txt' if update_var else '_only.txt')
  if mode.lower() =='train': 
    prior_mean = defaultdict(lambda :(np.array([[start_mean],[start_mean],[start_mean]],dtype = 'float64')))
    prior_var = defaultdict(lambda :(np.array([[var_1],[var_2],[var_3]],dtype = 'float64')))
  elif mode.lower() not in ['test','train']: 
    print(f'WARNING: your input of mode is {mode} and the current mode is "Train" ')

  with open(file_name,'w',encoding="utf-8") as f:
    str2 = 'game, current_winner, current_loser, winner_mean_old, loser_mean_old,  winner_mean, loser_mean, winner_std, loser_std, phat'
    f.write(str2+'\n')   
    count,tie,total,total_discrepancy = 0,0,0,0
    b= np.log(10)/400
    
    for game in np.arange(winners.shape[0]):
        current_winner = winners[game]
        current_loser = losers[game]
        # current surface [1,3]
        current_surface = surface[game]
        # current player
        current_player = [current_winner, current_loser]
        for player in current_player:
            if player not in prior_mean:
                prior_mean[player]
                prior_var[player]
        
        new_mean = prior_mean.copy()
        new_var = prior_var.copy()
        # winners & loser rating
        winner_mean = new_mean[current_winner]
        loser_mean = new_mean[current_loser]
        winner_var = new_var[current_winner]
        loser_var = new_var[current_loser]
        
        # calculate p(x)
        mu_w = current_surface.dot(winner_mean)[0] 
        mu_l = current_surface.dot(loser_mean)[0] 

        p_w = sigmoid(mu_w, mu_l) 
        p_l = 1-p_w
 
        # compare and count the correct prediction
        if mu_w > mu_l:
          count +=1
        if mu_w == mu_l:
          tie +=1
        total +=1
        
        # calculate discrepancy
        discrepancy = -np.log(p_w)
        total_discrepancy += discrepancy
        
        #Calculate C
        C = 1/(1 + (b**2)*p_w*p_l*(current_surface.dot(winner_var) + current_surface.dot(loser_var)))
        #Calculate K
        rho_mat = np.array([[1, rho_12, rho_13],
                  [rho_12, 1, rho_23],
                  [rho_13, rho_23, 1]])
        S_wi = np.sqrt(current_surface.dot(winner_var))
        S_li = np.sqrt(current_surface.dot(loser_var))
        cur_rho = current_surface.dot(rho_mat)
        K_w = S_wi*np.sqrt(winner_var).T*cur_rho.T*b*C
        K_l = S_li*np.sqrt(loser_var).T*cur_rho.T*b*C
        
        # update player mean
        winner_mean_new = winner_mean + K_w.T*(1-p_w)
        loser_mean_new = loser_mean + K_l.T*(-p_l)
        new_mean[current_winner] = winner_mean_new
        new_mean[current_loser] = loser_mean_new
        prior_mean = new_mean

        #Calculate new win prob and new C
        mu_w_new = current_surface.dot(winner_mean_new)[0] # float
        mu_l_new = current_surface.dot(loser_mean_new)[0] #float
        p_w_new = sigmoid(mu_w_new, mu_l_new)
        p_l_new = 1-p_w_new
        C_new = 1/(1 + (b**2)*p_w_new*p_l_new*(current_surface.dot(winner_var) + current_surface.dot(loser_var)))
        #Calculate L, L_w, L_l
        L = np.square(cur_rho)
        L_w = (b**2)*p_w_new*p_l_new*L*winner_var.T*C_new
        L_w = L_w/par4_i 
        L_l = (b**2)*p_w_new*p_l_new*L*loser_var.T*C_new
        L_l = L_l/par4_i
        # update player variance
        if update_var:
          tmp_w = winner_var*(1-L_w).T
          tmp_l = loser_var*(1-L_l).T
          new_var[current_winner] = np.array([[max(np.array([par3_i**2]),tmp_w[0])[0]],[max(np.array([par3_i**2]),tmp_w[1])[0]],[max(np.array([par3_i**2]),tmp_w[2])[0]]],dtype = 'float64')
          new_var[current_loser] = np.array([[max(np.array([par3_i**2]),tmp_l[0])[0]],[max(np.array([par3_i**2]),tmp_l[1])[0]],[max(np.array([par3_i**2]),tmp_l[2])[0]]],dtype = 'float64')         
          prior_var = new_var
        
        var_w_new = current_surface.dot(winner_var)[0] 
        var_l_new = current_surface.dot(loser_var)[0] 
   
        
        # save results
        str2 ='%2d, %s, %s, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f\n'%(game+1,current_winner,current_loser,mu_w,mu_l, mu_w_new, mu_l_new, np.sqrt(var_w_new), np.sqrt(var_l_new), p_w)
        f.write(str2) # save
        
    f.close()
    error = total-count-tie
    acc = count/(total-tie)
    return prior_mean, prior_var, acc, (total,count,tie), total_discrepancy/winners.shape[0]  
   

"""### Different Sigma0"""

print(kwarg)

file_name = 'accRate'
file_name +='_both.txt' if kwarg['update_var'] else '_only.txt'
print(file_name)

accTr=[]
accTe=[]
num=[]
discrepancy = []
with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s, %2s, %2s, %2s\n'%('sig','accTrain','accTest',"(n,a,t)",'neg-loglike'))
  for i in range(0, len(init_sig)):
     # print('sigma0 = ',init_sig[i])
     est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Train') 
     _, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var) 
     #print('update mu only and use surface',f'acc_rate:{np.round(acc_te, 4)}')
     accTr.append(acc_tr)
     accTe.append(acc_te) 
     num.append(num_te)
     discrepancy.append(discrep)
     str_w = f'{init_sig[i]:3d}, {acc_tr:.4f},{acc_te:.4f},{num_te},{discrep:.4f}'
     f.write(str_w+"\n") # save
f.close()   


acc_idx_tr_only = accTr.index(max(accTr))
acc_idx_te_only = accTe.index(max(accTe))
print('sigma with best accuracy in train: %d'%(init_sig[acc_idx_tr_only]))
print('sigma with best accuracy in test: %d'%(init_sig[acc_idx_te_only]))
disc_idx_tr_only = discrepancy.index(min(discrepancy))
print('sigma with smallest neg-loglikelihood in train: %d'%(init_sig[disc_idx_tr_only]))

est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, **kwarg, mode = 'Train')
_, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, init_sig[disc_idx_tr_only]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var)


kwarg['update_var']=True
print(kwarg)
file_name = 'accRate'
file_name +='_both.txt' if kwarg['update_var'] else '_only.txt'
print(file_name)

accTr1=[]
accTe1=[]
num1=[]
discrepancy1 = []
with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s, %2s, %2s, %2s\n'%('sig','accTrain','accTest',"(n,a,t)",'neg-loglike'))
  for i in range(0, len(init_sig)):
     #print('sigma0 = ',init_sig[i])
     est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Train') 
     _, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[i]**2, init_sig[i]**2, init_sig[i]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var) 
     #print('update mu & sigma and use surface',f'acc_rate:{np.round(acc_te, 4)}')
     accTr1.append(acc_tr)
     accTe1.append(acc_te) 
     num1.append(num_te)
     discrepancy1.append(discrep)
     str_w = f'{init_sig[i]:3d}, {acc_tr:.4f},{acc_te:.4f},{num_te},{discrep:.4f}'
     f.write(str_w+"\n") # save
f.close()

acc1_idx_tr_only = accTr1.index(max(accTr1))
acc1_idx_te_only = accTe1.index(max(accTe1))
print('sigma with best accuracy in train: %d'%(init_sig[acc1_idx_tr_only]))
print('sigma with best accuracy in test: %d'%(init_sig[acc1_idx_te_only]))
disc1_idx_tr_only = discrepancy1.index(min(discrepancy1))
print('sigma with smallest neg-loglikelihood in train: %d'%(init_sig[disc1_idx_tr_only]))

est_mean, est_var, acc_tr, _, discrep = calculate_ratings(winners, losers, surface, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, **kwarg, mode = 'Train')
_, _, acc_te, num_te,_ = calculate_ratings(winners_val, losers_val, surface_val, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, init_sig[disc1_idx_tr_only]**2, **kwarg, mode = 'Test', prior_mean = est_mean, prior_var = est_var)


file_name = 'errRate'
file_name +='.txt'
with open(file_name, 'w', encoding="utf-8") as f:  
  f.write('%3s\n'%('mean only'))
  f.write('%3s, %2s,%2s\n'%('sig','accTest','neg-loglike'))
  for i in range(0, len(init_sig)):
     str_w = f'{init_sig[i]:3d} & {accTe[i]:.4f} & {discrepancy[i]:.4f}\\\\'
     f.write(str_w+"\n") 
        
  f.write("\n")     
  f.write('%3s\n'%('both mu and sigma'))
  f.write('%3s, %2s,%2s\n'%('sig','accTest','neg-loglike'))
  for i in range(0, len(init_sig)):
     str_w = f'{init_sig[i]:3d} & {accTe1[i]:.4f} & {discrepancy1[i]:.4f}\\\\'
     f.write(str_w+"\n")   
        
  f.write("\n")     
  f.write('%3s\n'%('together'))
  f.write('%3s, %2s,%2s, %3s, %2s,%2s\n'%('sig','accTest','neg-loglike','sig','accTest','neg-loglike'))
  for i in range(0, len(init_sig)):
     str_w = f'{init_sig[i]:3d} & {accTe[i]:.4f} & {discrepancy[i]:.4f} & & & {init_sig[i]:3d} & {accTe1[i]:.4f} & {discrepancy1[i]:.4f}\\\\'
     f.write(str_w+"\n")       
f.close()   
