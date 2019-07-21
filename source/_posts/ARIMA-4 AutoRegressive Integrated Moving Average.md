---
title: ARIMA-4 AutoRegressive Integrated Moving Average
date: 2019-07-20 14:45:08
tags: time series models
categories: time series models
---

如果我们将自回归差分模型(differencing with autoregression)和移动平均模型(moving average model)做一个组合，就可以得到非周期性的ARIMA模型，这个模型是non-seasonal的。ARIMA是AutoRegressive Integrated Moving Average的缩写，这里integration其实是differencing的逆过程。注意这里ARIMA模型中的序列是做过差分的序列，所以预测器中既包含了延迟的(lagged)时间$y_t$，也包含了延迟的误差$\epsilon_t$。<!--more-->

## Non-seasonal ARIMA models

如下是它的公式。
$$
y_t^{'}=c+\phi_1y_{t-1}^{'}+\ldots+\phi_py_{t-p}^{'}+\theta_1\epsilon_{t-1}+\ldots+\theta_q\epsilon_{t-q}+\epsilon_t
$$
上式中的$y'_t$就表示做过差分的序列，而且可能不只是一次差分。我们把这个模型叫做$\text{ARIMA}(p,d,q)$，此处参数解释如下：
* $p=$order of the autoregressive part
* $d=$degree of first diffencing involved
* $q=$order of the moving average part

在自回归模型和移动平均模型中的stationary和invertibility条件对于ARIMA模型同样适用，下面是由ARIMA模型引申出的一些具体模型：
* White noise: ARIMA(0,0,0)
* Random walk: ARIMA(0,1,0) 即自回归和移动平均的系数都为0，只做一次差分$y_t = y_{t-1}+\epsilon_t$
* Random walk with drift: ARIMA(0,1,0) 即$y_t = c+y_{t-1}+\epsilon_t$
* Autoregression: ARIMA(p,0,0)
* Moving average: ARIMA(0,0,q)

如果我们把模型用backshift notation表示出来：
$$
\begin{aligned}
(1-\phi_1B-\ldots-\phi_pB^p)(1-B)^dy_t&=c+(1+\theta_1B+\ldots+\theta_qB^q)\epsilon_t \\\\
AR(p)*d \ \text{differences} &= MA(q)
\end{aligned}
$$

## Understanding ARIMA models
上述公式中的常数$c$和差分次数$d$对于模型影响很大，我们可以做如下分析：
* $c=0 \ \text{and} \ d=0$，长期预测值会趋向于0
* $c=0 \ \text{and} \ d=1$，长期预测值会趋向于非0常数
* $c=0 \ \text{and} \ d=2$，长期预测值会变成一条直线
* $c\neq0 \ \text{and} \ d=0$，长期预测值会趋向于数据的平均值
* $c\neq0 \ \text{and} \ d=0$，长期预测值会变成一条直线
* $c\neq0 \ \text{and} \ d=0$，长期预测值会变成二次抛物线
* 