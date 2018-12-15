#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 21:24:56 2018

@author: YaoyunZhang
"""

#-------------------------------------------Functions (RuiHao) ---------------------------------------------------
def GetPrice(tickers,start,end):

    import pandas as pd
    pd.core.common.is_list_like = pd.api.types.is_list_like
    from pandas_datareader import data as pdr
    import fix_yahoo_finance as yf
    yf.pdr_override()
    
    print('***Downloading data.***')
    price = pdr.get_data_yahoo(tickers, start=start, end=end) ["Adj Close"]

    # Saving Data
    print('***Saving Data.***')
    price.to_csv('price.csv')
    return price
#------------------------------ 
## VALIDATION
#------------------------------ 
#cal risk cantribution of each asset
def cal_risk_contribution(weights,cov_matrix):
    import numpy as np
    
    pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    mc = np.dot(cov_matrix, weights) / pf_volatility #marginal contribution
    rc = np.multiply(mc,weights) #risk_contribution by weight
    return rc #vector

def cal_sum_sq_error(weights,cov_matrix): # sum squard error bewteen target rc(all equal) and real rc (risk contribution) 
    import numpy as np
    
    pf_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    target_rc = np.ones(len(weights))*(pf_volatility/len(weights))
    current_rc = cal_risk_contribution(weights,cov_matrix)
    
    sse= sum(np.square(current_rc-target_rc))
    return sse
    
