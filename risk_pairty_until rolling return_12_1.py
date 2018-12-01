#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:55:24 2018

@author: jax
"""


import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr      # For download data
import matplotlib.pyplot as plt
import scipy.optimize as opt

''' (a) Download data and clean data '''
yf.pdr_override()
tickers = ['BSV','LQD','IGIB','SHY','PDP', 'VTI','IXN','IJH', 'IJR','EMB','EFA','EEM']
close = pd.DataFrame(columns = tickers)

for i in tickers:
    close[i] = pdr.get_data_yahoo(i,start = '2007-01-01',end = '2018-10-31')['Adj Close']

close.dropna(inplace = True)  # Check if there is NA value
#close.to_csv('/Users/jax/Downloads/study/803/finanal project/close.csv',sep=',')




#------------------------------------------EDIT PART BELOW----------------------------------------------------------------------

#cal risk cantribution of each asset
def cal_risk_contribution(weights,cov_matrix):
    pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    mc = np.dot(cov_matrix, weights) / pf_volatility #marginal contribution
    rc = np.multiply(mc,weights) #risk_contribution by weight
    return rc #vector

def cal_sum_sq_error(weights,cov_matrix): # sum squard error bewteen target rc(all equal) and real rc (risk contribution) 
    pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    target_rc = np.ones(len(weights))*(pf_volatility/len(weights))
    current_rc = cal_risk_contribution(weights,cov_matrix)
    
    sse= sum(np.square(current_rc-target_rc))
    return sse

#my part
def risk_parity(target_sigma, cov_matrix):
    def risk_contribution(weights,cov_matrix):
        pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        mc = np.dot(cov_matrix, weights) / pf_volatility #marginal contribution
        rc = np.multiply(mc,weights) #risk_contribution by weight
        return rc #vector

    def sum_sq_error(weights): # sum squard error bewteen target rc(all equal) and real rc (risk contribution) 
        pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
        target_rc = np.ones(len(weights))*(pf_volatility/len(weights))
        current_rc = risk_contribution(weights,cov_matrix)
    
        sse= sum(np.square(current_rc-target_rc))
        return sse
 
    def constraint1(weights):
        return np.sum(weights)-1.0
 
    def constraint2(weights): #using global variable sigma to set constraint of target pf vol
        pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        return pf_volatility-target_sigma
     

    def cal_portfolio_weight():
        initial_weight=np.ones(len(cov_matrix))
    
        cons=({'type': 'eq', 'fun': constraint1},{'type': 'eq', 'fun': constraint2})
        res = opt.minimize(sum_sq_error, initial_weight, method='SLSQP',constraints=cons)
        return res.x
    
    return cal_portfolio_weight()
    

#------------------------------------------EDIT PART ABOVE----------------------------------------------------------------------


'''

# optimazation
def risk_parity(sigma,cov):

    n = len(cov)
    def func(w):
        # w is the weight
        # cov is the covariance matrix
        par = []
        n = len(w)
        num = np.dot(cov, w)
        den = np.sqrt(np.dot(np.dot(w.T,cov),w))
        for i in range(n):
            par += [(w[i]-den**2/(num[i]*n))**2]
        return sum(par)

    def const1(w):
        # sigma the std of the portfolio
        return np.sqrt(np.dot(np.dot(w.T,cov),w))-sigma

    def const2(w):
        # the sum of weight equal to one
        return sum(w)-1

    x0=np.ones(n)
    res = opt.minimize(func, x0, method='SLSQP',constraints=({'type': 'eq', 'fun': const1},{'type': 'eq', 'fun': const2}))
    return res.x
'''

n_leverage = 2
sigma = 0.1
Ndays = 250


return1 = close / close.shift(1) -1
return1.dropna(inplace=True)

adjust_return = return1.iloc[:,0:4] * n_leverage
adjust_return = pd.concat([adjust_return,return1.iloc[:,4:13]],axis=1)

cov_matrix = adjust_return.cov()
corr_matrix = adjust_return.corr()
var = adjust_return.var()
annual_return = adjust_return.mean()*250
cov = np.array(cov_matrix)


test_window = 'month'
if test_window == 'month':
    month_close = close.resample('m',closed = 'right').last()
    
    
# Get rolling basis position
last_month = 'NA'

#----------------------------------------------------------------------------------------
weight = pd.DataFrame(columns = tickers)
risk_contribution = pd.DataFrame(columns = tickers)
sse = pd.DataFrame(columns =['SUM Sq Error'])
#----------------------------------------------------------------------------------------


for i in range(len(adjust_return)):
    date = adjust_return.index[i]   
    if i>Ndays and date.month != last_month:              # Adjust weight monthly    
            
        rolling_cov = adjust_return.iloc[i-Ndays:i,:].cov()
        current_weight = risk_parity(sigma,np.array(rolling_cov))   
        
        
        #----------------------------------------------------------------------------------------
        weight.loc[date] = current_weight  
        risk_contribution.loc[date] = cal_risk_contribution(current_weight,np.array(rolling_cov))
        sse.loc[date] = cal_sum_sq_error(current_weight,np.array(rolling_cov))
        #----------------------------------------------------------------------------------------
        

    last_month = date.month
    
def get_daily_return(weight):
    ret = pd.Series()
    k = 0
    this_month = weight.index[k].month
    for i in range(Ndays,len(adjust_return)):       
                
        date = adjust_return.index[i]        # Use the same weight for every month since adjust monthly
        
        if date>weight.index[0]:             # Start calculate return when strategy begin
            if date.month != this_month:
                k +=1
                this_month =date.month
            
            daily_return = sum(weight.iloc[k,:]*adjust_return.iloc[i,:])
            ret[date] = daily_return
    
    return ret

ret1 = get_daily_return(weight)
net_value = np.cumprod(ret1+1)