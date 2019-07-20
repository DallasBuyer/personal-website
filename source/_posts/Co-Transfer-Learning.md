---
title: Co-Transfer Learning Using Coupled Markov Chains
date: 2019-04-25 16:21:47
tags: transfer learning
categories: transfer learning
---

个人感觉这篇文章很有新意，扩展了传统的迁移学习场景，和归纳式迁移学习一样，假设目标域也有标签数据，但是它的创新点在于co-transfer，即进行了双向迁移，使得任何一个领域都可以辅助其他领域迁移。而作者实现co-transfer在于利用了两种关系，一种是Intra-relationship，这里可以直接在同一个特征空间内用距离度量来构造；另外一种是inter-relationships，这里很巧妙地借助了co-occurrence information，<!--more-->注意这里的co-occurrence和多视角学习中的instance-level co-occurrence并不相同。 之后将两种关系合并构造一个coupled Markov chain形成了一个联合转移概率图，之后通过向标签分布的概率转移来实现标签预测，这里有趣的是还引入了多标签和标签ranking的概念，有兴趣的同学也可以借鉴这种思想解决多标签分类的问题。


## Scene and Motivation

本文提出了一个叫作Co-Transfer Learning的算法，这里的Co-Transfer意为协同迁移。传统的迁移学习场景一般假设利用包含大量有标签数据的源域辅助只包含少量有标签数据的目标域数据进行建模，之后对目标域的无标签数据进行分类。显然传统的迁移学习是单向的，即源域辅助目标域。而本文的Co-Transfer的出发点是能不能作多向的迁移，即没有源域和目标域之分。假设同时有多个领域的数据，其中每个领域都包含一部分有标签数据和一部分无标签数据，本文的目标就是这些领域互相迁移。举个例子，假设现在有文本和图像两个领域的数据，每个领域的数据都包含有标签和无标签两种数据，本文想通过某种方式为这两个领域的数据建立某种桥梁，从而让图像领域的数据辅助文本领域数据的建模，同时文本领域的数据也可以辅助图像领域的建模。如果存在多个领域的话就是多向的，每个领域都可以辅助其它任何一个领域进行建模，从而形成了Co-Transfer的思想。



## Problem Definition

有了以上出发点，现在就要把问题数学化。首先定义多个领域，这里每个领域定义为一个样本空间和一个特征空间，且每个领域的样本空间和特征空间都不同。将第 $i$ 个样本空间表示为 $\mathcal{X^{(i)}}$，第 $i$ 个特征空间表示为 $\mathcal{Y^{(i)}}$，则第 $i$ 个样本空间的某个样本表示为 $x_k^{(i)}$，其对应的特征向量表示为 $\mathbf{y}_k^{(i)}$ 。且对于任意的 $i\neq i'$，有 $\mathcal{X^{(i)}}\neq \mathcal{X^{(i')}}$ 且 $\mathcal{Y^{(i)}}\neq \mathcal{Y^{(i')}} $，即每两个领域的样本空间和特征空间均不相同。每个领域都存在一个有标签样本集 $\mathcal{L^{(i)}}$ 和一个无标签样本集 $\mathcal{U^{(i)}}$。其中 $\mathcal{L^{(i)}}$ 可用式(1)表示，${\bar n_i}$ 第 $i$ 个领域有标签样本的个数。$\mathcal{U^{(i)}}$ 可用式(2)表示，$\hat n_i$ 表示第 $i$ 个领域无标签样本的个数。Co-Transfer Learning的目的就是使用所有领域中的有标签样本同时训练 $N$ 个不同的分类器来预测无标签测试样本的类别。
$$
\mathcal {L^{(i)}}=\{ x_k^{(i)},C_k^{(i)} \}_{k=1}^{\bar n_i}
$$

$$
\mathcal {U^{(i)}}=\{ x_k^{(i)} \}_{k=\bar n_i+1}^{\bar n_i+\hat n_i}
$$



## The Proposed Framework

本文所提方法将问题建模成一个所有样本的联合转移概率图(joint transition probability graph)模型，这样就可以利用topic-sensitive PageRank和random walk with restart的思想来解决co-transfer的问题了。其中transition probabilities可以通过intra-relationships和inter-relationships来构造，这里intra-relationships是指在同一个样本空间内所有样本的关系，它可以基于样本之间的affinity metric来构造，inter-relationships是指在不同样本空间的样本之间的关系，它可以基于样本之间的co-occurrence information来构造。未完待续...



## Intrarelationships and Interrelationships

式(3)展示出了所有 $N$ 个样本空间的所有样本(有标签和无标签)，第 $i$ 个样本空间的所有样本数为 $n_i=\bar n_i+\hat n_i$。为了简化表示，我们设 $n=\sum _{i=1}^Kn_i$。
$$
\underbrace{x_1^{(1)},\dots,x_{n_1}^{(1)}}_{\text{1st instance space}},\underbrace{x_1^{(2)},\dots,x_{n_2}^{(2)}}_{\text{2st instance space}},\dots,\underbrace{x_1^{(N)},\dots,x_{n_N}^{(N)}}_{\text{Nst instance space}}
$$

### Intra-relationship

首先我们定义在同一个样本空间内的所有样本之间的intra-relationships，在第 $i$ 个样本空间内的样本 $x_k^{(i)}$ 和 $x_l^{(i)}$ 之间的affinity metric定义为 $a_{k,l}^{(i,i)}$ ， $a_{k,l}^{(i,i)}$ 可以基于两个样本在特征空间的特征表示来计算，如式(4)。
$$
a_{k,l}^{(i,i)}=exp[\frac {-\Vert \mathbf y_k^{(i)}-\mathbf y_l^{(i)}\Vert_2}{2\sigma^2}]
$$
从而所有的 $a_{k,l}^{(i,i)}$ 可以组合形成矩阵 $\mathbf A^{(i,i)}$，其大小为 $n_i-\text{by}-n_i$，其中的每个元素都代表此样本空间内两个样本之间的intra-relationship。这里重点来了，我们根据矩阵 $\mathbf A^{(i,i)}$ 就可以构造一个Markov transition probability matrix $\mathbf P^{(i,i)}$，怎么构造那，可以对 $\mathbf A^{(i,i)}$ 的每一列进行归一化从而得到 $\mathbf P^{(i,i)}$，使得 $\mathbf P^{(i,i)}$ 的每一列元素的和都为1。因为这里矩阵 $\mathbf A^{(i,i)}$ 是对称的，所以得到的 $\mathbf P^{(i,i)}$ 不仅列实现了归一化，行也是归一化的。则矩阵  $\mathbf P^{(i,i)}$ 定义了在随机游走过程中，当前样本访问其他样本的概率。



### Inter-relationships

接下来我们定义不同的样本空间之间的inter-relationships，这里关系是基于**co-occurrence information**定义的。将第 $i$ 个样本空间中第 $k$ 个样本和第 $j$ 个样本空间的第 $l$ 个样本之间的关系定义为 $o_{k,l}^{(i,j)}$，则可以基于 $o_{k,l}^{(i,j)}$ 定义一个不同特征空间之间的关系矩阵 $\mathbf A^{(i,j)}$，其大小为 $n_i-\text{by}-n_j$。这里 $\mathbf A^{(i,j)}$ 不一定是对称的，但是一定存在  $[\mathbf A^{(i,j)}]_{(k,l)}=[\mathbf A^{(j,i)}]$，这里 $\mathbf A^{(j,i)}$ 是 $\mathbf A^{(i,j)}$ 的转置。同样我们把 $\mathbf A^{(i,j)}$ 针对列归一化，可以得到 $\mathbf P^{(i,j)}$，注意这里某些列的和可能为0，因为两个样本空间之间存在某些我们找不到co-occurrence information的样本。这种情况下，就令这一列的每个元素值都等于 $\frac 1 n_i$，从而得到了由一个样本空间到另一个样本空间的概率转移矩阵。



## Coupled Markov-Chain

由上述过程得到了所有样本空间中样本间的概率转移矩阵，因为有Intra-relationship和inter-relationships，把它们联合起来即为***coupled***，将两种概率转移矩阵联合起来可以构造成一个Markov Chain，即完成了我们最初的目的，就是为所有样本空间中的样本建立某种联系，从而为co-transfer做准备。

那具体怎么联合两种概率转移矩阵那，首先假设在时刻 $t$ 访问样本空间 $i$ 中的所有样本的概率构成一个概率分布向量(probability distribution vector)：$x^{(i)}(t)=[x_1^{(i)}(t),x_2^{(i)}(t),\dots,x_{n_i}^{(i)}(t)]^{\mathrm T}$，作为一个概率分布向量，它满足以下条件 $\sum_{k=1}^{n_i}x_k^{(i)}(t)=1$。在coupled Markov chain中，我们考虑从 $\lbrace x^{(i)}(t) \rbrace_{i=1}^N$ 到 $\lbrace  x^{(i)}(t) \rbrace_{i=1}^N$ 的一步转移概率，如式(6)所示：
$$
x^{(i)}(t+1)=\sum_{j=1}^N\lambda_{i,j}\mathbf P^{(i,j)}x^{(j)}(t),\quad i=1,2,\dots,N.
$$
上式则表示在 $t+1$ 时刻，访问第 $i$ 个样本空间中样本的概率，等于在 $t$ 时刻从所有样本空间中的样本的转移概率加权和。这里为了保证得到的 $x^{(i)}(t+1)$ 仍然是一个概率分布向量，令 $\sum_{j=1}^N\lambda_{i,j}=1$。此处 $\lambda_{i,j}$ 本质上是在控制从第  $i$ 个样本空间到第 $j$ 个样本空间迁移的知识量。

以矩阵的形式，可以把式(6)表示成式(7)的形式：
$$
x(t+1)=\mathbf P x(t),\quad
x(t)=\begin{pmatrix} x^{(1)}(t) \\\\ x^{(2)}(t) \\\\ \vdots \\\\ x^{(N)}(t) \end{pmatrix}, \quad
\mathbf P=\begin{pmatrix} 
\lambda_{1,1}\mathbf P^{(1,1)} & \lambda_{1,2}\mathbf P^{(1,2)} & \dots & \lambda_{1,N}\mathbf P^{(1,N)} \\\\ 
\lambda_{2,1}\mathbf P^{(2,1)} & \lambda_{2,2}\mathbf P^{(2,2)} & \dots & \lambda_{2,N}\mathbf P^{(2,N)} \\\\ 
\vdots & \vdots & \vdots & \vdots \\\\ 
\lambda_{N,1}\mathbf P^{(N,1)} & \lambda_{N,2}\mathbf P^{(N,2)} & \dots & \lambda_{N,2}\mathbf P^{(N,2)}
\end{pmatrix}
$$
此处求出的 $\mathbf P$ 即为联合转移概率图(joint transition probability graph)。



## Co-Transfer Learning

给定上述的联合转移概率图后，我们假设一个random walker从已知标签的样本出发，这里的样本就表示图中的nodes。这个random walker根据 $\mathbf P$ 重复迭代的访问它的neighborhood节点。在每个step中，它的return to training instances的概率是 $\alpha$，walker最终达到一个steady-state稳定态 $\mathbf U$，$\mathbf U$ 最终能给出标签的ranking来表示每个测试样本的标签集的重要性。将上述问题形式化：
$$
(1-\alpha)\mathbf{PU}+\alpha\mathbf Q=\mathbf U
$$
此处 $\mathbf Q $ ($n-\text {by}-c$) 是根据来自不用的样本空间中的所有训练数据构造的类标签的概率分布向量(probability distribution vector of the class labels)。这里的restart参数 $\alpha$ 用于控制 $\mathbf Q$ 对于最终的label ranking的重要性或者说影响。

因为 $\mathbf P$ 是已知的，$\mathbf U$ 是要求解的，所以现在重点就成了给定训练数据，怎么构造 $\mathbf Q$，我们用以下式子来构造这个标签概率分布矩阵。
$$
\begin{align}
\mathbf Q &=[\mathbf q_1,\mathbf q_2,\dots,\mathbf q_c],\\\\
\mathbf q_d &=[\mathbf q_d^{(1)},q_d^{(2)},\dots,q_d^{(N)}],\\\\
\mathbf q_d^{(i)} &=[q_{d,1}^{(i)},q_{d,2}^{(i)},\dots,q_{d,n_i}^{(i)}] 
\end{align}
$$
 上面式子中 $d=1,2,\dots,c,$，这里 $c$ 表示类别数，则 $q_d^{(i)}$ 表示在第 $i$ 个样本空间中每个样本属于 $d$ 类的概率，即这里的 $\mathbf q_d^{(i)}$ 是样本的概率分布向量，它是以如下方式由均匀分布得到的。
$$
q_{d,k}^{(i)}=\lbrace \begin{matrix} 
\frac1{l_d^{(i)}},&\text{if }d\in C_k^{(i)}, \\\\ 0, & \text{otherwise.}
\end{matrix}
$$
这里的 $q_{d,k}^{(i)}$ 表示样本空间 $i$ 中第 $k$ 个样本属于类别 $d$ 的概率，这里 $l_d^{(i)}$ 表示第 $i$ 个样本空间中属于类别 $d$ 的样本个数。

现在已知了 $\mathbf Q$ 和 $\mathbf P$，$\mathbf U$ 可以通过迭代求解 $\mathbf U(t)=(1-\alpha)\mathbf {PU}(t-1)$ 得到。则样本 $x_k^{(i)}$ 的标签可以通过矩阵 $\mathbf U$ 对应这个样本那一行概率向量来预测，即如下：
$$
l_k^{(i)}=[u_{k,1}^{(i)},u_{k,2}^{(i)},\dots,u_{k,c}^{(i)}]
$$
对于single class标签预测，取式(13)中最大的数值对应的标签就可以。对于multiclass标签预测，我们对这些概率进行一个ranking，取前 $d'$ 个即可。



## Experiments

实验部分，式(7)中的 $\lambda$ 是需要人为赋值的，另外还需要co-occurrence information对inter-relationships进行计算。



## Reference

Wu Q , Ng M K , Ye Y . Cotransfer Learning Using Coupled Markov Chains with Restart. *IEEE Intelligent Systems*, 2014, 29(4):26-33.

Wu Q , Ng M K , Ye Y . Co-Transfer Learning via Joint Transition Probability Graph Based Method. in *KDD*, 2012.





