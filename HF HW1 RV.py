"""
Created on Thu Jan 6 08:27:32 2022

@author: alexie

FIN 687
HW 1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import numpy as np
import statsmodels.formula.api as smf
# import statsmodels.formula.api as smf
# import statsmodels.api as sm

# Imports

# Part 1)

# Q1)

bitcoin_rets = pd.read_csv('C:/Users/alexi/Downloads/BTC_USD.csv')
# contains the bitcoin returns for the sample period

snp_prices = pd.read_csv('C:/Users/alexi/Downloads/SP500_rets.csv')
# contains the prices for the S&P 500 index for sample period

bitcoin_rets['Datetime'] = pd.to_datetime(bitcoin_rets['Date'])
bitcoin_rets = bitcoin_rets.set_index('Datetime')
del bitcoin_rets['Date']
# Setting date index

snp_prices['Datetime'] = pd.to_datetime(snp_prices['Date'])
snp_prices = snp_prices.set_index('Datetime')
del snp_prices['Date']

del bitcoin_rets['Currency']
# deleting str
bitcoin_rets = bitcoin_rets.drop(columns='Unnamed: 3')

daily_bitcoin = bitcoin_rets.pct_change()
daily_snp = snp_prices.pct_change()
# calculating daily returns


# print (daily_bitcoin)
# print (daily_snp)

# Q2)

monthly_bc = daily_bitcoin.resample('M').agg(lambda x: (x + 1).prod() - 1)
monthly_snp = daily_snp.resample('M').agg(lambda x: (x + 1).prod() - 1)
# calculating monthly returns

print (monthly_bc)
print (monthly_snp)

# Q3)

ff3_rets = pd.read_csv('C:/Users/alexi/Downloads/FF3_Monthly.csv')
# contains the famma french returns for sample period

ff3_rets['Datetime'] = pd.to_datetime(ff3_rets['Date'])
ff3_rets = ff3_rets.set_index('Datetime')
del ff3_rets['Date']

ff3_decimal = ff3_rets.div(100)
# converting percentages to decimals

# print (ff3_decimal)

# Q4)

merge = pd.merge(monthly_bc, monthly_snp, how = 'inner', left_index = True, right_index = True)
merge = pd.merge(merge, ff3_decimal, how = 'inner', left_index = True, right_index = True)
merge = merge.rename(columns = {'Closing Price': 'BTC Returns', 'Close': 'SnP Returns'})
# merging dfs to make one big df

# print (merge)

# Q5) Done automatically

# Part 2

# Q1)

merge['BTC Excess Returns'] = merge['BTC Returns'] - merge['RF']
merge['SnP Excess Returns'] = merge['SnP Returns'] - merge['RF']
# calculating excess returns by subtracting from risk free rates

# print (excess_bitcoin)
# print (excess_snp)

bc_mean = merge['BTC Excess Returns'].mean()
annual_bc_mean = bc_mean * 12
print (annual_bc_mean)

bc_std = merge['BTC Excess Returns'].std()
annual_bc_std = bc_std * np.sqrt(12)
print (annual_bc_std)

snp_mean = merge['SnP Excess Returns'].mean()
annual_snp_mean = snp_mean * 12
print (annual_snp_mean)

snp_std = merge['SnP Excess Returns'].std()
annual_snp_std = snp_std * np.sqrt(12)
print (annual_snp_std)

bc_sharpe = annual_bc_mean / annual_bc_std
print (bc_sharpe)

snp_sharpe = annual_snp_mean / annual_snp_std
print (snp_sharpe)

corr_coef = merge.corr(method = 'pearson')
print (corr_coef)

skew = merge.skew()
print (skew)

kurtosis = merge.kurt()
print (kurtosis)

# Q2)

bc_cummrets = (1 + merge['BTC Excess Returns']).cumprod() - 1
snp_cummrets = (1 + merge['SnP Excess Returns']).cumprod() -1
# Cummulative returns

# BTC Cumm rets 
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(bc_cummrets)
ax1.set_xlabel('Date')
ax1.set_ylabel("Cumulative Returns")
ax1.set_title("BTC Cumulative Returns")
plt.show();

# SnP Cumm rets
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
ax2.plot(snp_cummrets)
ax2.set_xlabel('Date')
ax2.set_ylabel("Cumulative Returns")
ax2.set_title("SnP Cumulative Returns")
plt.show();

# Q3) 

hist_bc = plt.hist(merge['BTC Excess Returns'])
plt.show();

# Q4)
# Scatter plot
scatter = plt.figure()
plt.scatter(merge['BTC Excess Returns'], merge['SnP Excess Returns'])
plt.show()

# Q5)

# Regression

merge.columns = merge.columns.str.replace(" ","_")

fitted = smf.ols("BTC_Excess_Returns ~ SnP_Excess_Returns", merge).fit()
print (fitted.summary())

# Q6)

merge.columns = merge.columns.str.replace("-","_")
ff_fitted = smf.ols("BTC_Excess_Returns ~ SMB + HML + Mkt_RF", merge).fit()
print (ff_fitted.summary())

# Part 3

# Q1)

HF_rets = pd.read_csv('C:/Users/alexi/Downloads/HFR.csv')

HF_rets['Datetime'] = pd.to_datetime(HF_rets['Date'])
HF_rets = HF_rets.set_index('Datetime')
del HF_rets['Date']

# Q2)

merge = merge.loc['2015-01-31 00:00:00': '2021-11-30 00:00:00']

merge['HFR'] = HF_rets

merge['HFR Excess'] = merge['HFR'] - merge['RF']
# Q3)

HFR_mean = merge['HFR Excess'].mean() * 12
print (HFR_mean)

HFR_std = merge['HFR Excess'].std() * np.sqrt(12)
print (HFR_std)

HFR_sharpe = HFR_mean / HFR_std
print (HFR_sharpe)

merge['HFR Excess'].quantile(.05)

skew = merge['HFR Excess'].skew()
print (skew)

kurtosis = merge['HFR Excess'].kurt()
print (kurtosis)

# Q4)
merge.columns = merge.columns.str.replace(" ","_")

fitted3 = smf.ols("HFR_Excess ~ BTC_Excess_Returns", merge).fit()
print (fitted3.summary())

predicted_val = fitted3.predict()
residual = merge["HFR_Excess"] - predicted_val
IR = fitted3.params[0] / residual.std()

merge["HFR_Pos"] = np.where(merge["BTC_Excess_Returns"] > 0, merge["BTC_Excess_Returns"], 0)\

# Q5) 

fitted4 = smf.ols("HFR_Excess ~ BTC_Excess_Returns + HFR_Pos", merge).fit()
print (fitted4.summary())

predicted_val = fitted4.predict()
residual = merge["HFR_Excess"] - predicted_val
IR = fitted4.params[0] / residual.std()




