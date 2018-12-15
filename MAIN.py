
#-------------------------------------------- Get Data & Parameters ------------------------------------------------

"""
1.0 Initialization
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Import User Defined Classes
import BackTest as bt
import Portfolio_Optimization as po
opt = po.Optimize()

# Import User Defined Functions
from functions import GetPrice
from functions import cal_risk_contribution
from functions import cal_sum_sq_error

"""
2.0 Get Raw Data
"""
# 2.1 Parameters

# Alphabetically Ordered Tickers & Weights
tickers = ['BSV','LQD','IGIB','IVV','PDP','IXN','EMB','IJH','IJR']
bonds = ['BSV','LQD','IGIB','EMB']
w_BlackRock = [0.25,0.03,0.04,0.08,0.15,0.24,0.04,0.12,0.05]
# sum(w_BlackRock) # weights check

st = datetime.datetime(2008,1,1) # Start Time
end = datetime.datetime(2015,10,31) # End Time
num = len(tickers)
Ndays = 252 # Number of days a year

# 2.2 Download Data
price = GetPrice(tickers,st,end)
ret = price.pct_change().dropna()
    
"""
3.0 Data Cleaning
"""
# Check for NA/Null values
#print('NaN values: \n', np.sum(price.isna(),axis=0))
print('\nMissing values: \n',np.sum(price.isnull(),axis=0))

"""
4.0 Data Analysis
"""
# 4.1 Plot All Price Data
plt.figure()
price.plot(figsize=(20,10))
plt.title('All Assets All Time Daily Price Plot',fontsize=20)
plt.xlabel('ETFs',fontsize=15)
plt.ylabel('Close Price',fontsize=15)
plt.show()
# plt.savefig('PriceOverview')

"""
5. Properties of Data
"""
# 5.1 Mean & Volatility
mu = np.mean(ret)*Ndays  # Annualized Mean Returns
vol = np.std(ret)*np.sqrt(Ndays)  # Annualized Volatilities

# 5.2 Leverage Bonds
n_levg = 3
ret[bonds] *= n_levg 

# 5.3 Leveraged Mean & Volatility
mu_levg = np.mean(ret)*Ndays  # Annualized Mean Returns(After leverage)
vol_levg = np.std(ret)*np.sqrt(Ndays)  # Annualized Volatilities(After leverage)



"""
6. Cross-Assets Properties of Data
"""
# Covariance & Correlations
covMat = ret.cov()
corrMat = ret.corr()
print('\nCovariance Matrix of Portfolio: \n',covMat)
print('\nCorrelation Matrix of Portfolio: \n',corrMat)

#---------------------------------------- Rolling & Running (Yaoyun) ------------------------------------------


"""
7. Calculate Optimal Portolio Weights/Positions
"""
# 7.1 Get 1-Yr-Rolling Portfolio Covariance Matrices
# ------------------------------------------------------------

# All Rolling Covariance Matrices
all_covMats = ret.rolling(Ndays).cov().dropna()

# Only Get the Last 1-Yr-Cov-Mat of Every Month
month_covMats = all_covMats.groupby(pd.Grouper(freq='M',level=0)).tail(num)

# Dates where we extract Covariance Matrix for Weights Calc (Monthly)
cov_dates = np.unique(month_covMats.index.get_level_values('Date'))[:-1] # we don't need the last covariance matrix

# All Dates (Daily) (1 Yr after beginning of data)    
all_dates = ret.iloc[Ndays-1:,].index

# Where Turnover Happens (should == #s Covariance Matrix is extracted == #s Weights Calculated )
toID = np.diff(np.array(all_dates.month))!=0      
toID = pd.Series(toID, index=all_dates[1:])

# 7.2 Safty Checks
if sum(toID)!=len(cov_dates):
    print("Error! Turnover times and numbers of covariance matrices obtained don't match.")


# 7.3 Calculate Rolling Portfolio Positions & Daily Returns 
# ----------------------------------------------------------
# 7.3.1 Initialize Some Variables to Store Results
ret1 = pd.Series() # Risk Parity Portfolio Returns
ret2 = pd.Series() # 60/40 Portfolio Returns
w_ = np.ones(num)/num
w_strat = w_

strat = input("Choose a strategy to compare with risk parity strategy (either Mean_Variance, or 60/40): ")
exp_ret = 0.1

weight = pd.DataFrame(columns = ret.columns) # Risk Parity Portfolio Weights
risk_cont = pd.DataFrame(columns = ret.columns) # Risk Parity Portfolio Risk Contributions
sse = pd.DataFrame(columns =['SUM Sq Error']) # Risk Parity Portfolio Optimization Sum of Squared Errors

k = 0 
for i in range(0,len(toID)):
    date = toID.index[i]

    if toID[i]==True: # Signal for 调仓
        
        # Current Covariance Matrix
        cov_ = np.array(month_covMats.loc[cov_dates[k]])*Ndays
        
        
        # 7.3.2 Calculate Portfolio Weights for Current Month:
        if strat == 'Mean_Variance':
            mu_ret = ret.rolling(Ndays).mean().loc[date] * Ndays
            w_strat = opt.mean_variance(mu_ret,exp_ret,cov_)
          
        elif strat == '60/40':
            w_strat = w_BlackRock
        else:
            print("Error! Your imput must be either Mean_Variance, or 60/40（type in the exact words).")
        
        sigma_ = np.sqrt(np.dot(np.dot(w_strat,cov_), w_strat))
        w_ = opt.risk_parity(sigma_,cov_) # Checked √
        
        
        # 7.3.3 Store Results
        weight.loc[date] = w_
        risk_cont.loc[date] = cal_risk_contribution(w_,cov_)
        sse.loc[date] = cal_sum_sq_error(w_,cov_) 
        
        k += 1
    
    # else: Everything is unchanged
    
    # 7.3.4 Calculate Daily Returns
    ret1[date] = np.dot(w_,ret.loc[date]) # Risk-Parity
    ret2[date] = np.dot(w_strat,ret.loc[date])
    
"""
8. Results Presentation
"""
# 8.1 Check Optimization Results
# look at variable: weight
# look at variable: risk_cont
# look at variable: sse

# 8.2 Present Daily Returns & Net Values
mkt_ret = price.IVV[all_dates[0]:].pct_change()
port_rets = pd.DataFrame({'Risk Parity':ret1, strat:ret2 , 'Market Index': mkt_ret}).dropna()
net_value = np.cumprod(port_rets+1)

# Net Value Plot:
plt.figure(figsize=(15,8))
net_value.plot()
plt.title('Backtest Net Values Comparison',fontsize=12)
plt.xlabel('Date',fontsize=8)
plt.ylabel('Net Value',fontsize=8)
plt.show()

# 8.3 Back Test Parameters
BT = bt.tests(port_rets,'Market Index',0.05)
print(BT.summary())
