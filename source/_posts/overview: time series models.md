---
title: Overview of Forecasting-Principles and Practice
date: 2019-07-18 14:56:13
tags: time series models
categories: time series models, forecasting
---

最近开始学习一些有关时间序列模型的内容，在此记录下来。此系列博客主要根据一本在线电子书的内容整理而来，为了目的是记录自己的学习历程，让自己的印象更加深刻。这是这本书的[网页连接](https://www.otexts.org/fpp)。
这本数的题目叫做Forecasting:Principles and Practice，原来自己接触的大部分内容都属于机器学习，无非就是两大任务Prediction和Regreesion，而这里的Forecasting本人理解更倾向于是对时间序列进行预测，即预测未来某个时候挥发生的事情，而不是像分类和回归一样针对某个样本判断其类别或者是取值。


## <!--more-->


## 1. What can be forecast?
其实forecast有很多场景，比如预测未来5年整个国家需要的用电量，这会决定国家对于发电厂等基础设备的投资建设；比如预测某个公司下周客服中心的呼叫量，这会决定这个公司应该招收多少客服人员；再比如预测股市的走势以决定怎么投资，预测电商网站的销量以决定需要怎么调整供应链等等。这些场景都有一个共性的问题就是它们所产生的数据具有时序性。而一个事件的可预测性一般由以下几点决定：

* 我们是否充分理解导致预测结果的因素
* 我们能够获取多少以往的数据
* 预测是否会影响我们想要预测的事情

比如我们要预测用电量的话应该比较容易，因为首先我们有大量的数据，其次我们很理解是什么因素导致了用电量的变化，比如气温，节假日，经济环境等。但是想要预测汇率可能就有些困难了，因为我们虽然有大量的数据，但是我们不知道有哪些确定性的因素会影响汇率的变化。在forecasting中，很重要的一点是我们要知道什么时间我们能够精准的预测，什么时候预测的结果比投硬币要好。好的预测模型通常能够捕获历史数据中的patterns和relationships，而不是简单的复制过去发生的事情。我这里的关键就是我们采用什么样的模型能够把历史数据中的一些噪声给去掉，而提取出那些有效的patterns和relationships，并且根据未来发生事情的factors或者partial pattern来预测它的最终结果。

## 2. Forecasting, planning and goals
其实forecasting在商业中是一个很常见的统计学任务，它可以为未来的产品规划，物流和人事，提供一个长期的战略计划。但是，在商业环境中forecasting一般完成的比较差劲而且还容易和另外两个内容混淆，就是planning和goals。

* Forecasting: 是用已知的信息去尽可能准确去预测未来，这些已知信息包含historical data和knowledge of any future events that might impact the forecasts(factors)。
* Goals: 目标是你想要它发生的事情，这里不涉及计划或者你的预测，只是你定的一个方向，能否实现，或者怎么实现都合目标无关。
* Planning: 而计划就是为Forecasting和Goals负责的，计划就涉及根据预测来采取合适的行动从而让你的目标达成。

Forecasting是管理的决策行动中很关键的一部分，它为决策提供了根据，一般来说预测分为三类
* Short-term forecasts: scheduling of personnel, production and transportation.
* Medium-term forecasts: future resource requirements, such as raw materials, hire personnel, or buy machinery and equipment.
* Long-term forecasts:strategic planning, including market opportunities, environmental factors and internal resources.

## 3. Determining what to forecast
我们做预测的时候首先要清楚的就是预测什么，当面对很多历史数据时，要先根据商业环境和需求明确预测目的

* every product line, or for groups of products?
* every sales outlet, or for outlets grouped by region, or only for total sales?
* weekly data, monthly data or annual data?

在确定的需要预测是什么之后紧接着的问题就是how frequent，我们需要每天更新预测数据吗，还是每周，每个月？在知道了这些之后才开始怎么预测的问题，就涉及到了methods和data。

## 4. Forecasting data and methods
预测方法的选择取决于可获取的数据，如果我们无法获取到数据，那么可以采用**qualitative forecasting(also known as judgmental forecasting)**方法。而当以下条件满足时我们可以采用**quantitative forecasting**方法。
* numerical information about the past is available;
* it is reasonabe to assume that some aspects of the past patterns will continue into the future.

有很多quantitative forecasting方法可以采用，一般来说大部分的quantitative prediction问题要么使用series data(collected at regular intervals over time)，要么使用cross-sectional data(collected at a single point in time)。而本书关注于time series domain。

### 4.1. Time series forecasting
下面是一些例子：
* Daily IBM stock prices
* Monthly rainfall
* Quarterly sales results for Amazon
* Annual Google profits

本书只关注于一些固定区间时间长度的时间序列，比如hourly, daily, weekly, monthly, quarterly和annually。不关注irregularly spaced time series。
最简单的时间序列预测方法就是仅仅根据以往的时序数据来预测未来的时序数据，而不去考虑导致最终结果的因素。常见的用于预测的时间序列模型包括**decomposition models, exponential smoothing models and ARIMA models**。

### 4.2. Predictor variables and time series forecasting
一般来说，预测变量(predictor variables)在时间序列预测中很有效，假设我们现在想预测某个区域夏季一个小时的供电量，可以采用以下模型。
$$
\mathrm{ED} = f(\mathrm{current\ temperature, strength\ of\  economy, population, time\ of\ day, day\ of\ week, error}).
$$
其实上述关系也不完全正确，因为供电量中总有一些变化是无法用上述预测变量来解释的，而最右侧的$\mathrm{error}$项就可以代表随机波动和那些没有被包含到模型中的其他相关变量。我们可以叫上述为**expanatory model**，因为它可以帮助解释是什么引起了供电量需求的变化。
因为随着时间变化供电量可以形成一个时间序列，所以我们也可以用一个如下的**time series model**来预测。
$$
\mathrm{ED_{t+1}}=f(\mathrm{ED_t, ED_{t-1},ED_{t-2}, ED_{t-3}, \ldots,error})
$$
这里$t$表示当前一个小时，$t+1$表示下一个小时，$t-1$表示前一个小时。这里只采用了过去的时序数据，而没有外部变量。同样的，这里的$\mathrm{error}$表示一些随机波动和没有包含到模型里面的相关变量。当然也有第三种模型结合前两种，可以叫做混合模型。
$$
\mathrm{ED_{t+1}}=f(\mathrm{ED_t, current\ temperature, strength\ of\  economy, population, time\ of\ day, day\ of\ week, error})
$$
这类模型也叫做**dynamic regression model, panel data models, longitudinal models, transfer functions models, linear system models**。

## 5. The basic steps in a forecasting task

1. Problem definition
2. Gathering information
3. Preliminary(exploratory analysis)
4. Choosing and fitting models
5. Using and evaluating a forecasting model