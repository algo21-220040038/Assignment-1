# Assignment-1
A factor strategy of Convertible bond.
Based on the reports of some broker's analyst, I combine the impilied volatility and volatility of stock and obtain 
an effective factor to trade convertible bond in China security market.

The data set is a .mat fileincluding all daily trading data of stock and convbond obtained from Wind. 
The Implied Volatility.py is used to compute the implied volatility of option value in convbond. 
![image](https://raw.githubusercontent.com/algo21-220040038/Assignment-1/master/result/Volatility.png)

This is a picture of implied volatility of convbond and volatility of stock. (精测电子)

Here shows the final result of backtest from 2016 to 2021/02.

![image](https://raw.githubusercontent.com/algo21-220040038/Assignment-1/master/result/Portfolio-return.png)

Portfolio 1, 2, 3 are obtained from top 33%, 33%~66% and 66%~100% all tradable convbond storted by the implied volatility factor.
