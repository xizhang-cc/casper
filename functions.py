
import numpy as np
import pandas as pd
from functools import partial
from numba import jit
import random
from itertools import combinations
from scipy.special import factorial
from matplotlib import pyplot as plt
import copy

#==============================================================================
#================phi(t) = m^kappa * alpha * exp(-beta*t)=====================
#==================b(t) = 0 ===================================================
#======================== marks: g(m) = =======================================
#==============================================================================


def marked_exp_mean_count(t, alpha, beta, kappa, gamma, tc, obs_events):
    
    if t <= tc:
        value = len([e for e in obs_events if e[0] <= t])
    else: 

        n_alpha = gamma*beta
        
        temp_df = pd.DataFrame(obs_events, columns=['event_time', 'event_mark'])
        temp_df['value'] = temp_df.apply(lambda x: np.power(x['event_mark'], kappa)*\
                                         np.exp(-beta*(tc - x['event_time'])) ,axis=1)
        temp_val = temp_df['value'].sum() * alpha / (gamma*beta)
        
        if n_alpha == beta: 
            f1 = n_alpha*(t-tc)
        else:
            f1 = ( 1 - np.exp(-(beta-n_alpha)*(t-tc)) ) *gamma / (1-gamma)
        
        value = temp_val*f1 + len(obs_events)
            
    return value

def marked_generation_count_mean(t, k, alpha, beta, kappa, gamma, tc, init_obs):
      
    if t <= tc:
        value = 0
    else:
        n_alpha = gamma*beta
        
        temp_df = pd.DataFrame(init_obs, columns=['event_time', 'event_mark'])
        temp_df['value'] = temp_df.apply(lambda x: np.power(x['event_mark'], kappa)* np.exp(-beta*(tc - x['event_time'])) ,axis=1)
        temp_val = temp_df['value'].sum() * alpha / (gamma*beta)
        
    
        td = t-tc
        temp_sum = 0
        for j in np.arange(k+1):
            temp_sum = temp_sum + np.power(beta*td, j) * np.exp(-beta*td)/ factorial(j)
    
        value = temp_val * np.power((n_alpha/beta), k+1) * (1-temp_sum)
    
    return value

def marked_count_mean_approximation(t, maxG, alpha, beta, kappa, gamma, tc, init_obs):
    
    if t <= tc:
        value = len([e for e in init_obs if e[0] <= t])
    else: 
        value = len(init_obs)

        for k in np.arange(0, maxG):
            value = value + marked_generation_count_mean(t, k, alpha, beta, kappa, gamma, tc, init_obs)
            
    return value

def marked_generation_count_var(t, k, alpha, beta, kappa, gamma, nv, tc, init_obs):
    
    if t <= tc:
        value = 0.0
        
    else:
        temp_df = pd.DataFrame(init_obs, columns=['event_time', 'event_mark'])
        temp_df['value'] = temp_df.apply(lambda x: np.power(x['event_mark'], kappa)* np.exp(-beta*(tc - x['event_time'])) ,axis=1)
        temp_val = temp_df['value'].sum() 
                
        eta = temp_val * (alpha / beta) 
              
        lk = marked_generation_count_mean(t, k, alpha, beta, kappa ,gamma, tc, init_obs)
                          
        if gamma == 1:
            value = lk + np.power(lk,2) * (k-1) / eta
        else:
            value = lk + np.power(lk,2) * nv *\
                ( 1-np.power(gamma,k) )/(np.power(gamma,k-1) - np.power(gamma,k))/ (eta * np.power(gamma,2))
    return value



def marked_count_var_approximation(t, maxG, alpha, beta, kappa, gamma, nv, tc, init_obs):
    if t <= tc:
        var_approx = 0.0
    else:
        var_approx = 0.0
            
        count_mean = marked_exp_mean_count(t, alpha, beta, kappa, gamma, tc, init_obs)
        
        for k in np.arange(0, maxG):
            
            mean_k =  marked_generation_count_mean(t, k, alpha, beta, kappa, gamma, tc, init_obs)
            var_k = marked_generation_count_var(t, k, alpha, beta, kappa, gamma, nv, tc, init_obs)
            
            temp = sum([marked_generation_count_mean(t, i, alpha, beta, kappa, gamma, tc, init_obs) \
                        for i in np.arange(0, k+1)]) + len(init_obs)
            
            if mean_k == 0:
                para = 1   
            else:
                para = 1+ 2*(count_mean - temp)/mean_k
        
            var_approx = var_approx + para*var_k
            
    return var_approx 

#===============================================================================
#===============================================================================
#===============================================================================


def sec2scale(time_str):
    
    granularity_hours_scale_setup = {'10min': 2, '30min': 6, '1h': 9, '2h': 19}

    try:
        granularity_hours = granularity_hours_scale_setup[time_str]
    except: 
        granularity_hours = 24
    
    gs = 60*60*granularity_hours
    
    return gs

def get_mark_dist(realization, tc, include_tc = True):
    # find current empirical mark distribution
    if include_tc == True:
        obs_events = [e for e in realization['hw'] if e[0]<= tc]
    else:
        obs_events = [e for e in realization['hw'] if e[0]< tc]

    obs_df = pd.DataFrame(obs_events, columns=['event_time', 'event_mark'])    
    obs_df['mark_count'] =  obs_df.groupby('event_mark')['event_mark'].transform('count') 
    mark_dist = obs_df[['event_mark', 'mark_count']].drop_duplicates()
    mark_dist['mark_prob'] = mark_dist['mark_count'] /  mark_dist['mark_count'].sum()
    
    mark_dist_df = mark_dist[['event_mark', 'mark_prob']]
    mark_dist_array = np.array(mark_dist_df)
    
    return mark_dist_array, mark_dist_df

@jit(nopython=True)
def p_func(gamma, beta, tij):
    
    if gamma == 1:
        value = gamma * beta*tij
    else:
        value = (1 - np.exp(-beta*(1-gamma)*tij))*gamma / (1-gamma) 
    return value

@jit(nopython=True)
def p_grad_cal(gamma, beta, tij):
    if tij<0:
        print('ERROR')
        
    if gamma == 1:
        pd_gamma = beta*tij
        pd_beta = gamma*tij
    else:
        pd_gamma = (1-np.exp(-beta*(1-gamma)*tij)) / np.power(1-gamma, 2) -\
                    gamma * beta * tij * np.exp(-beta*(1-gamma)*tij) / (1-gamma) 
                    
        pd_beta = gamma*tij*np.exp(-beta*(1-gamma)*tij)
        
    return pd_gamma, pd_beta

        
@jit(nopython=True)
def q_func(kappa, beta, ti, i_hist, mark_dist):
        
    a = np.dot(i_hist[:,1]**kappa, np.exp(-beta* np.subtract(ti,  i_hist[:,0])))
    f1 = np.sum((mark_dist[:,0]**kappa)*mark_dist[:,1])
        
    return a / f1

@jit(nopython=True)
def q_grad_cal(kappa, beta, ti, i_hist, mark_dist):
    
    i_hist_times = i_hist[:,0]
    i_hist_marks = i_hist[:,1]

    f1 = np.sum( np.power(mark_dist[:,0], kappa)*mark_dist[:,1])
    
    fd_1 = np.sum( np.log(mark_dist[:,0]) * np.power(mark_dist[:,0], kappa)*mark_dist[:,1] )
        
    qd_kappa_1 = np.dot(np.log(i_hist_marks)*np.power(i_hist_marks, kappa) , \
                        np.exp(-beta* np.subtract(ti, i_hist_times)) ) / f1   
        
    qd_kappa_2 = np.dot(np.power(i_hist_marks, kappa), \
                        np.exp(-beta* np.subtract(ti, i_hist_times))) * fd_1 / np.power(f1,2)
        
    qd_kappa = qd_kappa_1 - qd_kappa_2

    qd_beta = - np.dot(np.power(i_hist_marks, kappa), np.subtract(ti, i_hist_times) * \
                       np.exp(-beta* np.subtract(ti, i_hist_times))) / f1
    
    return qd_kappa, qd_beta

def param_update(theta, step_size, grad_theta, theta_min, theta_max):
    
    theta_next = theta - grad_theta * step_size
    
    theta_next = np.clip(theta_next, theta_min, theta_max)
            
    return theta_next

#==============================================================================
#==============================================================================
#==============================================================================

@jit(nopython=True)
def obj_cal(theta, tc, obs_events ,mark_dist, time_pairs): 
        
    value = 0
    
    gamma = theta[0]
    beta = theta[1]
    kappa = theta[2]
    
    for ti in np.unique(time_pairs[:,0]):  
                    
        i_hist = obs_events[obs_events[:,0] <= ti]
        
        N_ti = len(i_hist) 
    
        qi_value = q_func(kappa, beta, ti, i_hist, mark_dist)
        
        tj_list = time_pairs[time_pairs[:,0]==ti][:,1]
    
        for tj in tj_list:
                        
            N_tj = len(obs_events[obs_events[:,0] <= tj])
            
            pij_value = p_func(gamma, beta, (tj-ti)) 
       
            value = value + (pij_value * qi_value -(N_tj-N_ti))**2
            
    value = value / len(time_pairs)
    
    return value

@jit(nopython=True)
def gradient_cal(theta, tc, obs_events, mark_dist, time_pairs):


    gamma = theta[0]
    beta = theta[1]
    kappa = theta[2]
    
    gamma_grad = 0
    beta_grad = 0
    kappa_grad = 0
    
    for ti in np.unique(time_pairs[:,0]):  

        i_hist = obs_events[obs_events[:,0] <= ti] #[e for e in obs_events if e[0]<=ti]
        N_ti = len(i_hist) 
        
        qi_value = q_func(kappa, beta, ti, i_hist, mark_dist)
        qi_grad_kappa, qi_grad_beta = q_grad_cal(kappa, beta, ti, i_hist, mark_dist)

        tj_list = time_pairs[time_pairs[:,0]==ti][:,1]

        for tj in tj_list:
            
            N_tj = len(obs_events[obs_events[:,0] <= tj])
            
            pij_value = p_func(gamma, beta, tj-ti)
            
            pij_grad_gamma, pij_grad_beta = p_grad_cal(gamma, beta, tj-ti)
            
            gamma_grad = gamma_grad + 2*(pij_value * qi_value - (N_tj-N_ti)) * qi_value * pij_grad_gamma
                              
            kappa_grad = kappa_grad + 2*(pij_value * qi_value - (N_tj-N_ti)) * pij_value * qi_grad_kappa
                               
            beta_grad = beta_grad + 2*(pij_value * qi_value - (N_tj-N_ti)) *\
                              (pij_value * qi_grad_beta + qi_value* pij_grad_beta)
                                                     
    return np.array([gamma_grad/len(time_pairs), beta_grad/len(time_pairs), kappa_grad/len(time_pairs)])




def projected_gradient_descent(realization, tc, theta_init, theta_min, theta_max, time_pairs = None,\
                                              step_size_init=0.1, bt=0.7, max_iter = 20000, epsilon=1e-6):
    


    theta_init = np.array(theta_init)

    events = np.array(realization['hw'])
    obs_events = events[events[:,0]<=tc]
    
    if time_pairs is None:

        grid_times = np.append(obs_events[:,0], tc) 
        time_pairs = np.array(list(combinations(grid_times, 2)))
    
    
    # find current empirical mark distribution
    mark_dist, _ = get_mark_dist(realization, tc)

    update_func = partial(param_update, theta_min = np.array(theta_min), theta_max=np.array(theta_max))
    obj_func = partial(obj_cal, tc=tc, obs_events=obs_events, mark_dist=mark_dist, time_pairs=time_pairs)
    grad_func = partial(gradient_cal, tc=tc, obs_events= obs_events, mark_dist=mark_dist, time_pairs=time_pairs)       

    
    step_size_list = [step_size_init]
    obj_list = [obj_func(theta_init)]
    
    # initialization
    step_size = step_size_init
    theta_curr = theta_init
    norm_diff = epsilon + 1
    iter_c = 0
    
    while (iter_c < max_iter) and (norm_diff > epsilon):
    
        step_size = step_size_init
        
        grad_curr = grad_func(theta_curr)
        theta_next = update_func(theta_curr, step_size, grad_curr)
        
        next_obj =  obj_func(theta_next)
        inter_count = 0
        while (obj_list[-1] - next_obj <= 0):
            step_size = step_size * bt
            theta_next = update_func(theta_curr, step_size, grad_curr)
            next_obj = obj_func(theta_next)
            
            inter_count = inter_count + 1
            if inter_count > 100:
                theta_next = theta_curr
                break
        
        norm_diff = np.linalg.norm( theta_next - theta_curr)
        iter_c = iter_c + 1
        
        obj_list.append(next_obj)
        step_size_list.append(step_size)
        
        theta_curr = theta_next
                            
        
    theta_learned = theta_curr
    theta_learned = pd.Series(theta_learned, index=['gamma', 'beta', 'kappa'])
    
    f1 = np.sum( np.power(mark_dist[:,0], theta_learned[2])*mark_dist[:,1])
    learned_alpha = theta_learned[0] * theta_learned[1] / f1     
    
    theta_learned['alpha'] = learned_alpha
    
    return theta_learned, obj_list




def learn(censoring_time, cascade, theta_min, theta_max, theta_init, method = 'censored'):
    # scaling factor to covert event time in seconds to desired time granularity
    gs = sec2scale(censoring_time)  
    
    theta_min = pd.Series(theta_min )
    theta_max = pd.Series(theta_max)
    theta_init = pd.Series(theta_init)
    
    data = copy.deepcopy(cascade)
    data['hw'] = [(e[0]/gs, e[1]) for e in data['hw'] ]
    
    tc = pd.Timedelta(censoring_time).total_seconds() / gs
    events = np.array(data['hw'])
    obs_events = events[events[:,0]<=tc]
    
    if method == 'all':     
        # all pairs
        grid_times = np.append(obs_events[:,0], tc) 
        time_pairs = np.array(list(combinations(grid_times, 2)))
        print(len(grid_times))
        print(len(time_pairs))
    elif method == 'censored':
        # only predict at tc
        time_pairs = np.column_stack((obs_events[:,0], tc* np.ones(len(obs_events))))
        
    elif method == 'fix_num':
        N = 20
        temp_1 = np.column_stack((obs_events[:,0], tc* np.ones(len(obs_events))))
        temp_2 = np.array(list(combinations(obs_events[:,0], 2)))
        
        if (len(temp_1) + len(temp_2) < N):
            time_pairs =  np.concatenate((temp_1, temp_2), axis=0)
        else:
            if len(obs_events) < N:
                make_ups = np.random.choice(len(temp_2), size= N-len(temp_1), replace=False)
                time_pairs = np.concatenate((temp_1, temp_2[make_ups,:]), axis=0)
            else:
                time_pairs = temp_1[np.random.choice(len(temp_1), size= N, replace=False), :]
    
    
    theta_learned, obj_list = projected_gradient_descent(data, tc,\
                  np.array(theta_init), np.array(theta_min), np.array(theta_max),\
                  time_pairs = time_pairs, step_size_init = 1e-2)

    
    return theta_learned




# used for getting results for a single cascades
def get_prediction_results(realization, theta_learned, tc, pred_times, get_bounds= False, N_max = 10,  eb = 0.05 ):
    
    # tc: right-censored time, observed events up to and include this time.
    # pts: list of times at which the learned and gt are being compared

    
    gamma_learned = theta_learned['gamma']
    beta_learned = theta_learned['beta']
    kappa_learned = theta_learned['kappa']
    alpha_learned = theta_learned['alpha']
    

    events = realization['hw']
    
    
    # ts = pts[0]
    obs_events = [e for e in realization['hw'] if e[0]<= tc]
    
    s_df = pd.DataFrame({'pred_time': pred_times})
    # 
    
    s_df['true_count'] = s_df.apply(lambda x: len([e for e in events if e[0] <= x['pred_time']]), axis=1)
    
    print('calculating conditional means')
    s_df['pred_count'] = s_df.apply(lambda x: marked_exp_mean_count(x['pred_time'], alpha_learned,\
                            beta_learned, kappa_learned, gamma_learned, tc, obs_events ) , axis=1)   
        
        
    if get_bounds:
        
        mark_dist, _ = get_mark_dist(realization, tc, include_tc = True)
        
        # f1 = np.sum( np.power(mark_dist[:,0], kappa_learned)*mark_dist[:,1])
        # learned_gamma = theta_learned['alpha'] * f1 / theta_learned['beta']
        
        f2 = np.sum( np.power(mark_dist[:,0], 2*kappa_learned)*mark_dist[:,1])
        nv = np.power(alpha_learned/beta_learned, 2) * f2

        print('calculating conditional variances')

        s_df['pred_var'] = s_df.apply(lambda x: marked_count_var_approximation(x['pred_time'], N_max, alpha_learned,\
                                beta_learned, kappa_learned, gamma_learned, nv, tc, obs_events ) , axis=1)   
    
                
        s_df['lower_bound']  = s_df.apply(lambda x: x['pred_count'] -  np.sqrt( ( x['pred_var'] / eb ) ) if x['pred_time'] > tc else x['true_count'] ,axis=1)
           
        s_df['lower_bound'] =  s_df['lower_bound'].mask( (s_df['lower_bound'] <len(obs_events)) & (s_df['pred_time']>tc), len(obs_events))

         
        s_df['upper_bound']  = s_df.apply(lambda x: x['pred_count'] + np.sqrt( (x['pred_var'] / eb ))  if x['pred_time'] > tc else x['true_count'] ,axis=1)

    return s_df

        

def predict(delta_times, cascade, censoring_time, theta_learned, get_bounds=False):
        
    # scaling factor to covert event time in seconds to desired time granularity
    gs = sec2scale(censoring_time)  
    
    data = copy.deepcopy(cascade)
    data['hw'] = [(e[0]/gs, e[1]) for e in data['hw'] ]
    
    tc = pd.Timedelta(censoring_time).total_seconds() / gs
    
    if type(delta_times) == str:
        delta_times = [delta_times]
    
    delta_t = [pd.Timedelta(delta_time).total_seconds() / gs for delta_time in delta_times] 
    
    pred_times = np.array(delta_t) + tc   
    r_df = get_prediction_results(data, theta_learned, tc, pred_times, get_bounds= get_bounds, eb = 0.05)
    
    results_df = r_df.drop(columns=['pred_time'])
    results_df['delta_time'] = delta_times
    results_df = results_df.set_index('delta_time')
    
    pred_vals = list(results_df.pred_count)
    
    if len(pred_vals) == 1:
        pred_vals = pred_vals[0]
    
    return pred_vals



def visulization_plot(figName, cascade, censoring_time, theta_learned, last_pred_time ='7d', get_bounds=False, fontsize=12):
    data = copy.deepcopy(cascade)
    
    # scaling factor to covert event time in seconds to desired time granularity
    gs = sec2scale(censoring_time)  
    
    
    data['hw'] = [(e[0]/gs, e[1]) for e in data['hw'] ]
    
    tc = pd.Timedelta(censoring_time).total_seconds() / gs
    t_end = pd.Timedelta(last_pred_time).total_seconds() / gs
    
    pts_1 = np.linspace(0, tc, 10)
    pts_2 = np.linspace(tc, t_end, 40)
    pts = np.concatenate([pts_1, pts_2])
    
    s_df = get_prediction_results(data, theta_learned, tc, pts, get_bounds= get_bounds, eb = 0.05)
    
    
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(s_df['pred_time'], s_df['pred_count'], 'r', label='predicted count')
    ax.plot(s_df['pred_time'], s_df['true_count'], 'g', label='true count')    

    ax.set_xticks([tc, t_end])
    ax.set_xticklabels(['tc={}'.format(censoring_time), last_pred_time ], fontsize=fontsize)
    ax.legend(loc='lower right',fontsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)
    ax.set_ylabel('count',fontsize=fontsize)
    if get_bounds:
        ax.fill_between(s_df['pred_time'], s_df['lower_bound'] , s_df['upper_bound'],facecolor='lightgrey')
        ax.set_title('Predicted Counts with 95% Intervals', fontsize=fontsize)
    else:
        ax.set_title('Predicted Counts', fontsize=fontsize)
    
    return s_df

