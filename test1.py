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
import datetime
from pandas_datareader import data as pdr      # For download data
import scipy.optimize as opt

''' (a) Download data and clean data '''
yf.pdr_override()
tickers = ['BSV','LQD','IVV','PDP']
close = pd.DataFrame(columns = tickers)

st = datetime.datetime(2007,1,1)
end = datetime.datetime(2015,10,31)
close = pdr.get_data_yahoo(tickers,start = st,end = end)["Adj Close"]

close.dropna(inplace = True)  # Check if there is NA value

return1 = close / close.shift(1) -1
return1.dropna(inplace=True)
# 3 times leverage for bond ETFs
n_leverage = 3
#return1[['BSV','LQD','IGIB','SHY','EMB']] *= n_leverage    # leverage for bond ETF
adjust_return = return1

cov_matrix = adjust_return.cov()
corr_matrix = adjust_return.corr()
var = adjust_return.var()
annual_return = adjust_return.mean()*250

#compare weight
BlackRock_weight = [0.22,0.32,0.3,0.16]
BlackRock_adjust = [0.22,0.12,0.3,0.16]

# portfolio variance
cov = np.array(cov_matrix)

# optimazation
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
        
        initial_weight=np.ones(len(cov_matrix))/12
        cons=({'type': 'eq', 'fun': constraint1},{'type': 'eq', 'fun': constraint2})
        res = opt.minimize(sum_sq_error, initial_weight, method='SLSQP',constraints=cons)
        return res.x
    
    return cal_portfolio_weight()

# optimazation
test_window = 'month'
if test_window == 'month':
    month_close = close.resample('m',closed = 'right').last()
    
    

Ndays = 250

# Get rolling basis position
last_month = 'NA'
weight = pd.DataFrame(columns = tickers)
weight = pd.DataFrame(columns = tickers)
risk_contribution = pd.DataFrame(columns = tickers)
sse = pd.DataFrame(columns =['SUM Sq Error'])
current_weight = np.ones(len(tickers))

for i in range(len(adjust_return)):
    date = adjust_return.index[i]   
    if i>Ndays and date.month != last_month:              # Adjust weight monthly          
        
        rolling_cov = adjust_return.iloc[i-Ndays:i,:].cov()    
        rolling_sigma = np.dot(BlackRock_adjust,np.dot(rolling_cov,BlackRock_adjust))
        
        #----------------------------------------------------------------
        current_weight= risk_parity(rolling_sigma,np.array(rolling_cov)) 
        weight.loc[date] = current_weight
        
        risk_contribution.loc[date] = cal_risk_contribution(current_weight,np.array(rolling_cov))
        sse.loc[date] = cal_sum_sq_error(current_weight,np.array(rolling_cov))
        #---------------------------------------------------------------------------------------------------
        

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
net_value.plot()
weight_4060 = pd.DataFrame([BlackRock_adjust]*len(weight),index = weight.index,columns = weight.columns)

ret2 = get_daily_return(weight_4060)
net_value2 = np.cumprod(ret2+1)
net_value2.plot()

items = ['sharpe','annual_return','annual_vol','market_beta','','']
backtest = pd.DataFrame(index=items,columns = ['risk_parity','40-60'])
sharpe = [ret1.mean()/ret1.std()*np.sqrt(250),ret2.mean()/ret2.std()*np.sqrt(250)]
annual_return = []