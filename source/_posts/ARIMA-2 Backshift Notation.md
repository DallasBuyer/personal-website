---
title: ARIMA-2 Backshift Notation
date: 2019-07-19 14:27:38
tags: time series models
categories: time series models
---

在研究时间序列延迟的时候，后移符号(backshift notation)非常有用，可以用$B$表示后移操作。也有一些文献使用$L$表示lag而不是$B$表示backshift。换句话说，$B$在$y_t$上面的操作 就是将时序数据后移了一个周期，当然两个$B$就是后移了两个周期。<!--more-->

$$
\begin{aligned}
By_t & = y_{t-1} \\\\
B(By_t) & = B^2y_t=y_{t-2}
\end{aligned}
$$

对于monthly数据来说，如果我们想要表示去年同一个月的数据，就可以写成$B^{12}y_t=y_{t-12}$。后移操作对于描述differencing过程也非常的方便，如下。
$$
y^{'}_t=y_t-y_{t-1}=y_t-By_t=(1-B)y_t
$$
所以由上述式子注意到一阶差分可以表示成$(1-B)$，同样的，二阶差分可以表示成如下形式。
$$
\begin{aligned}
y_t^{''} & =y_t-2y_{t-1}+y_{t-2} \\\\
& =(1-2B+B)y_t \\\\
& =(1-B)^2y_t
\end{aligned}
$$
所以一般来说，一个$dth-order$的差分可以写成如下形式。
$$
(1-B)^dy_t
$$
后移操作在组合不同的差分计算是非常有用，它可以被当作一个适合于常用代数规则的运算符号。比如，当一个周期性差分紧跟着一个一阶差分时可以表示成如下形式。
$$
\begin{aligned}
(1-B)(1-B^{m})y_t & =(1-B-B^m+B^{m+1})y_t \\\\
                  & =y_t-y_{t-1}-y_{t-m}+y_{t-m-1}
\end{aligned}
$$