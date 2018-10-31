---
layout: post
title: Gaussian Process Regresssion
data: 2018-08-01 12:00:00 
category: 
tags:
---

Such a nice name for such a neat tool! Gaussian process regression (GPR) is a powerful method for regression, i.e. for predicting continuous valued outputs. But, before we go more into it, there are some concepts you should be familiar with if you want to fully appreciate and understand GPR. In particular, these concepts are the multivariate Gaussian distribution and Bayesian linear regression. I’ll briefly talk about them and go over the key results, but I can’t possibly substitute for a more thorough reading and  education of these topics. So, if you feel like looking more into them at other resources, I would think you wise! 

The main mathematical structure behind GPR is the _multivariate Gaussian distribution_. Multivariate Gaussians are simply generalizations of univariate Gaussian distributions to $$n$$ dimensions. The probability density function (PDF) of a set of random variables $$ x \in {\rm I\!R}^n $$ that are distributed by a multivariate Gaussian with mean $$ \mu \in {\rm I\!R}^n $$ and positive semi-definite (note: positive semidefinite means it has non-negative eigenvalues) covariance $$ \Sigma \in {\rm I\!R}^{n \times n} $$ is:
<p> $$p(x; \mu, \Sigma) = \dfrac{1}{\sqrt{ (2\pi)^n |\Sigma|}} \mathrm{exp} ( - \dfrac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) ) $$ </p>

An important important property of multivariate Gaussians that we’ll need for GP regression is the _conditioning property_. Specifically, if we write our random vector $$ x \sim \mathcal{N}( \mu, \Sigma ) \in {\rm I\!R}^{n} $$ as

$$ x_a =
\begin{bmatrix}
x_a \\
x_b
\end{bmatrix} $$
where
$$ x_a =
\begin{bmatrix}
x_1 \\
. \\
. \\
x_k
\end{bmatrix} $$ & $$ x_b =
\begin{bmatrix}
x_{k+1} \\
. \\
. \\
x_n
\end{bmatrix} $$
, then 
$$ \mu_a =
\begin{bmatrix}
\mu_a \\
\mu_b
\end{bmatrix}$$
& 
$$\Sigma = 
\begin{bmatrix}
\Sigma_{aa} & \Sigma_{ab} \\
\Sigma_{ba} & \Sigma_{bb}
\end{bmatrix} $$.

The conditioning property then says that the distribution of $$x_a$$ given (conditional on) $$x_b$$ is also multivariate Gaussian:

<p> $$ x_a | x_b \sim \mathcal{N}( \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b),$$ $$\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$$ </p> 

This is a very neat result and one that we’ll find good use for later! So just remember this for now or come back to it if you forget.


