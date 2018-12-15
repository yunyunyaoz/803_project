#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:57:34 2018

@author: YaoyunZhang
"""

class Optimize:
    
    def __init__(self):
        self.np = __import__('numpy')
        self.sc = __import__('scipy.optimize')
        self.w = 0

    def risk_parity(self, target_vol, covMat):

        def cal_portfolio_weight():
            init_w = self.np.ones(len(covMat))/len(covMat)
    
            cons=({'type': 'eq', 'fun': constraint1}, {'type': 'eq', 'fun': constraint2})
            res = self.sc.optimize.minimize(sum_sq_error, init_w, method='SLSQP',constraints=cons)
            return res.x
    
        def risk_contribution(weights,covMat):
            pf_volatility = self.np.sqrt(self.np.dot(weights, self.np.dot(covMat, weights)))
            mc = self.np.dot(covMat, weights) / pf_volatility #marginal contribution
            rc = self.np.multiply(mc,weights) #risk_contribution by weight
            return rc #vector
    
        def sum_sq_error(weights): # sum squard error bewteen target rc(all equal) and real rc (risk contribution) 
            pf_volatility = self.np.sqrt(self.np.dot(weights, self.np.dot(covMat, weights)))
    
            target_rc = self.np.ones(len(weights))*(pf_volatility/len(weights))
            current_rc = risk_contribution(weights,covMat)
    
            sse= sum(self.np.square(current_rc - target_rc))
            return sse 
 
        def constraint1(weights):
            return self.np.sum(weights)-1.0
 
        def constraint2(weights): #using global variable sigma to set constraint of target pf vol
            pf_volatility = self.np.sqrt(self.np.dot(weights, self.np.dot(covMat, weights)))
            return pf_volatility-target_vol
     
        return cal_portfolio_weight()
    

    def mean_variance(self,r,r_goal,cov):
        #r is the return of each etfs
        #r_goal is the goal return in the constraint
        #cov is the covariance matrix of each etfs
        n = len(cov)
        def func(w):
            # w is the weight
            # cov is the covariance matrix
            return self.np.sqrt(self.np.dot(self.np.dot(w.T,cov),w))

        def const1(w):
            # sigma the std of the portfolio
            return self.np.dot(r.T,w)-r_goal

        def const2(w):
            # the sum of weight equal to one
            return sum(w)-1

        x0 = self.np.ones(n)
        res = self.sc.optimize.minimize(func, x0, method='SLSQP',constraints=({'type': 'eq', 'fun': const1},{'type': 'eq', 'fun': const2}))
        return res.x    

