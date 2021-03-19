# Assignment-1-A factor strategy of Convertible bond.

Convertible bonds have a trading history of more than 20 years in China. However, it has only gradually received the attention of investors, especially institutional investors, in recent years. Here is a graph showing the volume changes of my country's convertible bonds in the past two decades.

![image](https://raw.githubusercontent.com/algo21-220040038/Assignment-1/master/result/Amount.png)

Based on the reports of some broker's analyst and derivatives' characteristic, I combine the impilied volatility with volatility of stock and obtain 
an effective factor to trade convertible bonds in China security market.

The data set is a .mat file including all daily trading data of stock and some special data of convertible bonds (like strbvalue, convprice) obtained from Wind. See details in CONVBOND.mat数据说明v1.pdf.

The Implied Volatility.py is used to compute the implied volatility of option value in convertible bonds, and good factor is produced by subtracting stock volatility from implied volatility of convertible bonds.
This is a picture of implied volatility of convbond and volatility of stock. (精测电子)

![image](https://raw.githubusercontent.com/algo21-220040038/Assignment-1/master/result/Volatility.png)

To illustrate the performance of this factor, I used the convertible bonds that were tradable on the market in the past 4 years to conduct a backtest. According to the weight of the convertible bond balance, the backtest selects the top 33%, 33~66% and the last 34% of convertible bonds according to the ranking of the factor.
The picture below shows the final result from 2016 to 2021/02.

![image](https://raw.githubusercontent.com/algo21-220040038/Assignment-1/master/result/Portfolio-return.png)

Portfolio 1, 2, 3 are obtained from top 33%, 33%~66% and 66%~100% tradable convertible bonds, respectively.
