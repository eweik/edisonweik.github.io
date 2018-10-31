---
layout: post
title: Gaussian Process Regresssion
data: 2018-08-01 12:00:00 
category: 
tags:
---

Such a nice name for such a neat tool! 

Gaussian process regression (GPR) is a powerful method for regression, i.e. for predicting continuous valued outputs. But, before we go more into it, there are some concepts you should be familiar with if you want to fully appreciate and understand GPR. In particular, these concepts are the multivariate Gaussian distribution and Bayesian linear regression. I’ll briefly talk about them and go over the key results, but I can’t possibly substitute for a more thorough reading and  education of these topics. So, if you feel like looking more into them at other resources, I would think you wise! 

The main mathematical structure behind GPR is the **multivariate Gaussian distribution**. Multivariate Gaussians are simply generalizations of univariate Gaussian distributions to $$n$$ dimensions. The probability density function (PDF) of a set of random variables $$ x \in {\rm I\!R}^n $$ that are distributed by a multivariate Gaussian with mean $$ \mu \in {\rm I\!R}^n $$ and positive semi-definite (note: positive semidefinite means it has non-negative eigenvalues) covariance $$ \Sigma \in {\rm I\!R}^{n \times n} $$ is:
<p> $$p(x; \mu, \Sigma) = \dfrac{1}{\sqrt{ (2\pi)^n |\Sigma|}} \mathrm{exp} ( - \dfrac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) ) $$ </p>

An important important property of multivariate Gaussians that we’ll need for GP regression is the _conditioning property_. Specifically, if we write our random vector $$ x \sim \mathcal{N}( \mu, \Sigma ) \in {\rm I\!R}^{n} $$ as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ x = \begin{bmatrix} x_a \\ x_b \end{bmatrix} $$ where $$ x_a = \begin{bmatrix} x_1 \\ . \\ . \\ x_k \end{bmatrix} $$ and $$ x_b = \begin{bmatrix} x_{k+1} \\ . \\ . \\ x_n \end{bmatrix} $$, then

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  $$ \mu = \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}$$ and $$\Sigma =  \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix} $$.

The conditioning property then says that the distribution of $$x_a$$ given (conditional on) $$x_b$$ is also multivariate Gaussian:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$x_a | x_b \sim \mathcal{N}( \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b),$$ $$\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$$ 

This is a very neat result and one that we’ll find good use for later! So just remember this for now or come back to it if you forget.



The next tool we’ll need in our arsenal is **Bayesian linear regression**. Without getting too involved in the details of Bayesian linear regression, some important details about BLR is that you can _use your prior knowledge_ of the dataset to help with your predictions and you _get a posterior distribution of the prediction_!

This is in contrast to other methods which return a single value with no account of the possible uncertainty in that value (these other methods use _Maximum Likelihood estimation_). Bayesian linear regression, however, uses _Maximum a Posteriori estimation_. Figure 1 shows the difference between these two methods.

# Gaussian Processes
The final tool we need to know before we get to GP regression are Gaussian processes (GPs)!

**Gaussian processes** are defined as a set of random variables $$ \{ f(x) : x \in X \} $$, indexed by elements $$ x $$ from some index set $$ X $$, such that any finite subset of this set $$ \{ f(x_1),...,f(x_n) \} $$ is multivariate Gaussian distributed! Simple enough, right?

Fortunately, another, perhaps more intuitive, way to think of Gaussian Processes are as distributions over random functions! This distribution is specified by a mean function $$ m( \dot{} ) $$ and a covariance function $$ k( \dot{}, \dot{} ) $$, which we denotes as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ f( \dot{} ) \sim \mathcal{GP} ( m( \dot{} ) $$ $$ k( \dot{}, \dot{} ) ) $$ 

Just for  preciseness, $$ m( \dot{} ) $$ must be a real function and $$ k( \dot{}, \dot{} ) $$ must be a valid kernel function.
