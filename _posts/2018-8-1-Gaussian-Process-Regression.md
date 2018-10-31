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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
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

Fortunately, another, perhaps more intuitive, way to think of Gaussian Processes are as distributions over random functions! This distribution is specified by a mean function $$ m( \cdot ) $$ and a covariance function $$ k( \cdot, \cdot ) $$, which we denotes as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ f( \cdot ) \sim \mathcal{GP} ( m( \cdot ) $$ $$ k( \cdot, \cdot ) ) $$ 

Just for  preciseness, $$ m( \cdot ) $$ must be a real function and $$ k( \cdot, \cdot ) $$ must be a valid kernel function.

To make this a bit more clear, let’s consider the Normal Distribution $$ \mathcal{N} ( \mu,$$ $$ \sigma^2) $$. When we sample a number $$ x \sim \mathcal{N} (0, 1) $$, the probability distribution for the possible values of $$ x $$ is just a standard bell curve. But, when we sample $$ x \sim \mathcal{N} (0, 10) $$, then probability distribution for values of $$ x $$ is a much wider and shorter shaped bell curve (see figure 2). If you play around with this more, you’ll begin to notice that the shape of the normal distribution is ultimately determined by the variation parameter $$ \sigma^2 $$. The larger $$ \sigma^2 $$ is, the wider the distribution is and the more likely it is that we’ll sample a number that is not close to 0.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/normal.png" width="600">
</p>
_Figure 2_: Probability distribution for a Gaussian distribution with variance 1 on the left and variance 10 on the right.

<br>

In a similar manner, we can sample a function from a Gaussian Process. And, just like when we sample a number from a Normal distribution, when we sample a function from a GP, the distribution of the types of functions that we are likely to get is ultimately determined by the kernel or covariance function $$ k( \cdot, \cdot ) $$. Some examples of types of kernels are:

* constant kernel: $$ k(x_i, x_j) = C $$

* linear kernel: $$ k(x_i, x_j) = x_i \cdot x_j $$

* squared exponential kernel: $$ k(x_i, x_j) = \mathrm{exp} ( -\dfrac{1}{2l^2} | x_i - x_j |^2 ) $$

* symmetric kernel: $$ k(x_i, x_j) = \mathrm{exp} (- ( \mathrm{min}( |x_i - x_j|, |x_i + x_j| ) )^2 ) $$

* periodic kernel: $$ k(t_i, t_j) = \mathrm{exp} ( - \mathrm{sin}^2 ( \alpha \pi (t_i - t_j))) $$

So, we can sample different types of functions from a Gaussian Process defined by each of these kernels. In this post, I’ll only look at Gaussian Processes with a zero mean function, i.e. $$ m(\cdot) = 0 $$ , and you should be able to see the effect this has on the functions we get from the examples below and how varying $$ m(\cdot) $$ would affect the types of functions sampled. Note: in each of the figures I show a picture with one function sampled from the Gaussian Process and another picture showing ten functions sampled from the GP.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_number.png" width="600">
</p>
_Figure 3_: Functions sampled from a Gaussian Process with a constant kernel $$ k(x_i, x_j) = 1 $$. In this case, the functions we sample are just constant numbers, but notice how the numbers center around 0 and vary in both directions. This is just like sampling a number from Gaussian distribution $$ x \sim \mathcal{N} (0, 1) $$, except here the numbers are functions!

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_line.png" width="600">
</p>
_Figure 4_: The functions sampled from a Gaussian Process with a linear kernel $$ k(x_i, x_j) = x_i \cdot x_j $$ are just linear functions. Notice, in the picture on the right, that at each point $$ x $$ the values $$ f(x) $$ center at 0 and vary in proportion to $$ x $$. That is, $$ f(x) $$ at $$ x = 4 $$ varies much more than at $$ x = 1 $$. In fact, the distribution of $$ f(x) $$ is distributed at $$ x $$ according to $$ x \sim \mathcal{N} (0, t) $$.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_se.png" width="600">
</p>
_Figure 5_: The functions sampled from a Gaussian Process with a squared exponential kernel $$ k(x_i, x_j) = \mathrm{exp}( -\dfrac{1}{2 l^2}| x_i - x_j |^2 ) $$ are very smooth functions. In fact, they’re infinitely differential at every point. In this figure, $$ l=1 $$, but changing $$ l $$ would result in either smoother (with higher $$ l $$) or more volative (with smaller $$ l $$) functions.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_symmetric.png" width="600">
</p>
_Figure 6_: The functions sampled from a Gaussian Process with the kernel $$ k(x_i, x_j) = \mathrm{exp} ( - \alpha ( \mathrm{min}( |x_i - x_j|, |x_i + x_j| ) )^2) $$ are symmetric functions about $$ t = 0 $$.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_periodic.png" width="600">
</p>
_Figure 7_: The functions sampled from a Gaussian Process with the kernel $$ k(x_i, x_j) = \mathrm{exp} ( - \mathrm{sin}^2 ( \alpha \pi (x_i - x_j) ) ) $$  are periodic functions.

<br>

