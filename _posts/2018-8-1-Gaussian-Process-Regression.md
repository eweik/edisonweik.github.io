---
layout: post
title: Gaussian Process Regresssion
data: 2018-08-01 12:00:00 
category: 
tags:
---

Such a big sounding name! 

I remember the first time I heard about Gaussian Processes. It was the summer of 2016, while I was at CERN, and another student was working on a project using Gaussian Processes. I remember him sounding very smart talking about "kernel functions" and other stuff and thinking I could never understand that. Well, here I am - two years later. And with a better understanding of Gaussian Processes and  regression. Hopefully it all makes sense to you.

Gaussian process regression (GPR) is general method for predicting continuous valued outputs. It's very powerful, stronger than normal linear regression and support vector machines. But, before we go more into it, there are some concepts you should be familiar with if you want to fully appreciate and understand GPR. In particular, you should know about the multivariate Gaussian distribution and Bayesian linear regression. I’ll briefly talk about them right now, but I can’t possibly substitute for a more thorough reading of these topics.

The main mathematical structure behind GPR is the **multivariate Gaussian distribution**. The multivariate Gaussian distributions is simply an extension of the univariate Gaussian distribution to $$n$$ dimensions. If a univariate Gaussian desribes one random variable, then a multivariate Gaussian describes an entire vector of random variables. However, it doesn't just describe the behavior of each component, is also describes how the components vary with each other.

The probability density function (PDF) of a set of random variables $$ x \in {\rm I\!R}^n $$ that are distributed by a multivariate Gaussian with mean $$ \mu \in {\rm I\!R}^n $$ and positive semi-definite (i.e. non-negative eigenvalues) covariance $$ \Sigma \in {\rm I\!R}^{n \times n} $$ is:
<p> $$p(x; \mu, \Sigma) = \dfrac{1}{\sqrt{ (2\pi)^n |\Sigma|}} \mathrm{exp} ( - \dfrac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) ) $$ </p>

One important important property of multivariate Gaussians that we’ll need for GP regression is the _conditioning property_ and the _summation property_. If we write our random vector $$ x \sim \mathcal{N}( \mu, \Sigma ) \in {\rm I\!R}^{n} $$ as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ x = \begin{bmatrix} x_a \\ x_b \end{bmatrix} $$ where $$ x_a = \begin{bmatrix} x_1 \\ . \\ . \\ x_k \end{bmatrix} $$ and $$ x_b = \begin{bmatrix} x_{k+1} \\ . \\ . \\ x_n \end{bmatrix} $$, then

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  $$ \mu = \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}$$ and $$\Sigma =  \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix} $$.

The conditioning property says that the distribution of $$x_a$$ given (conditional on) $$x_b$$ is also multivariate Gaussian! It's distribution is given by

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$x_a | x_b \sim \mathcal{N}( \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b),$$ $$\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$$,

The summation property says that if $$ x \sim \mathcal{N}( \mu_x, \Sigma_x ) $$ and $$ y \sim \mathcal{N}( \mu_y, \Sigma_y ) $$, then 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ x + y \sim \mathcal{N}( \mu_x + \mu_y, \Sigma_x + \Sigma_y ) $$

These are a very neat results! So just remember this for now or come back to it if you forget.

<br>

The next concept that's useful is **Bayesian linear regression**. Without getting too involved in the details of Bayesian linear regression, some important points about BLR is that you can _use your prior knowledge_ of the dataset to help your predictions and you _get a posterior distribution of the prediction_! This is in contrast to other methods which return a single value with no account of the possible uncertainty in that value (these other methods use _Maximum Likelihood estimation_). Bayesian linear regression uses _Maximum a Posteriori estimation_ for the full distribution of the output, called the posterior distribution. Figure 1 shows the difference between these two methods.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/regression1.png" width="600">
</p>
_Figure 1_: Bayesian and Classical (Frequentist) predictions on linear observations. Both are very similar, but one advantage of the Bayesian approach is the uncertainty we get in our prediction. In the figure, the shaded region represents 2 standard deviations above and below the prediction.

<br>

# Gaussian Processes
Just before we get to Gaussian Process regression, it's obviously important to understand Gaussian processes (GPs).

**Gaussian processes** are defined as a set of random variables $$ \{ f(x) : x \in X \} $$, indexed by elements $$ x $$ from some index set $$ X $$, such that any finite subset of this set $$ \{ f(x_1),...,f(x_n) \} $$ is multivariate Gaussian distributed! An intuitive way to think of them are as infinite dimensional extensions of the multivariate Gaussian distribution. Okay, maybe that's not so intuitive. It's actually quite hard for me to think of multivariate Gaussians in 2 or 3 dimensions, and I can't even imagine what they're like in infinite dimensions. Fortunately, when we consider only a finite subset of them, then we can't treat as multivariate Gaussians:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ f( \cdot ) \sim \mathcal{N} ( 0 $$ $$ k( x, x ) ) $$ 

Thinking of Gaussian Processes this way allows us see them as distributions over random functions! This distribution is specified by a mean function $$ m( \cdot ) $$ and a covariance function $$ k( \cdot, \cdot ) $$. So another way to denote $$ f(x) $$ is as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ f( \cdot ) \sim \mathcal{GP} ( m( \cdot ) $$ $$ k( \cdot, \cdot ) ) $$ 

Just forpreciseness, $$ m( \cdot ) $$ must be a real function and $$ k( \cdot, \cdot ) $$ must be a valid kernel function.

One way I like to think about them is by first considering the Normal Distribution $$ \mathcal{N} ( \mu,$$ $$ \sigma^2) $$. When we sample a number $$ x \sim \mathcal{N} (0, 1) $$, the probability distribution for the possible values of $$ x $$ is just a standard bell curve. But, when we sample $$ x \sim \mathcal{N} (0, 10) $$, then probability distribution for values of $$ x $$ is a much wider and shorter shaped bell curve (see figure 2). If you play around with this more, you’ll begin to notice that the shape of the normal distribution is ultimately determined by the variation parameter $$ \sigma^2 $$. The larger $$ \sigma^2 $$ is, the wider the distribution is and the more likely it is that we’ll sample a number that is not close to 0.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/normal.png" width="600">
</p>
_Figure 2_: Probability distribution for a Gaussian distribution with variance 1 on the left and variance 10 on the right.

<br>

In a similar manner, we can sample a function from a Gaussian Process. And, just like when we sample a number from a Normal distribution, when we sample a function from a GP, the distribution of the types of functions that we are likely to get is ultimately determined by the kernel or covariance function $$ k( \cdot, \cdot ) $$. 

In this post, I’ll only look at Gaussian Processes with a zero mean function, i.e. $$ m(\cdot) = 0 $$ , and you should be able to see the effect this has on the functions we get from the examples below and how varying $$ m(\cdot) $$ would affect the types of functions sampled. In each of the figures, I show a picture with one function sampled from the Gaussian Process and another showing twenty functions sampled from the GP.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_number.png" width="600">
</p>
_Figure 3_: Functions sampled from a Gaussian Process with a constant kernel $$ k(x_i, x_j) = 1 $$. In this case, the functions we sample are just constant numbers, but notice how the numbers center around 0 and vary in both directions. This is just like sampling a number from Gaussian distribution $$ x \sim \mathcal{N} (0, 1) $$, except here the numbers are functions!

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_line.png" width="600">
</p>
_Figure 4_: The functions sampled from a Gaussian Process with a linear kernel $$ k(x_i, x_j) = x_i \cdot x_j $$ are just linear functions. Notice, in the picture on the right, that at each point $$ x $$ the values $$ f(x) $$ center at 0 and vary in proportion to $$ x $$. That is, $$ f(x) $$ at $$ x = 4 $$ varies much more than at $$ x = 1 $$. In fact, the distribution of $$ f(x) $$ is distributed at $$ x $$ according to $$ x \sim \mathcal{N} (0, t) $$.

<br>

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

Another point I forgot to make, and which I didn’t show in the figures, is that different hyperparameters are associated with each of these kernels can effect the types of functions sampled. For example, in the periodic kernel $$ k = \mathrm{exp} ( - \mathrm{sin}^2 ( \alpha \pi (x_i - x_j) ) ) $$ in figure 7, making $$ \alpha $$ larger would result in functions with much higher frequencies of oscillation, while smaller $$ \alpha $$ would result in functions with lower frequencies. These artifacts of hyperparameters are present in all the kernel functions shown above.

<br>

# Gaussian Process Regression
Finally, we arrive at Gaussian Process Regression. In these types of problems we are given a dataset $$ \{({\bf x}_{i} , y_{i}) |i=1,...,m\} $$, where $$ y = f({\bf x}) + \epsilon $$ is a noisy observation of the underlying function $$ f({\bf x}) $$. It is important to assume that the noise is $$ \epsilon \sim \mathcal{N}(0,\sigma^2) $$.

The goal is to predict the $$y$$ values for future $$x$$’s. If we knew what the function $$ f({\bf x}) $$ was then we wouldn’t need to do GP regression because we could just plug in our future ’s and get a perfect prediction. Unfortunately, we don’t know $$ f({\bf x}) $$. So, what we’ll do is sample some functions from a Gaussian Process and go from there.

In theory, we can sample an infinite number of functions and choose only the ones that fit our data. But, in practice this is obviously not feasible. So, if we write down our model again

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ y = f({\bf x}) + \epsilon $$ where $$ f( {\bf x}) \sim \mathcal{GP}(0, k(\cdot, \cdot)) $$ and $$ \epsilon \sim \mathcal{N}(0,\sigma^2) $$,

then we’ll notice that because $$ f({\bf x}) $$ is multivariate Gaussian distributed (from the definition of Gaussian Processes) and $$ \epsilon $$ is Gaussian distributed (by assumption), then $$y$$ must also be multivariate Gaussian distributed, i.e. $$ y \sim \mathcal{N}(0, k(\dot{},\dot{}) + \sigma^2) $$! 

If we have our labeled dataset $$ \{({\bf x}_{i} , y_{i}) |i=1,...,m\} $$ and we also have the unlabeled dataset $$ \{ {\bf x}_{i} |i=m+1,...,n\} $$
for which we want to predict , then a reasonable assumption is that both  and  come from the same distribution, namely

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ \begin{bmatrix} y \\ y_* \end{bmatrix} \sim \mathcal{N}( 0, \begin{bmatrix} k(x,x) + \sigma^2I && k(x,x_*) \\ k(x_*, x) && k(x_*, x_*) + \sigma^2 I \end{bmatrix} ) $$.

where $$ x $$ denotes labeled data and $$ x_* $$ denotes unlabeled data. So, from the conditioning property of multivariate Gaussians, we can get our conditional distribution of $$ y_* $$ given $$ x $$ as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ {\bf y}_*|X, {\bf y}, X_* \sim \mathcal{N}( \mu_*, \Sigma_*) $$

where

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ \mu_* = k_*[k + \sigma_n^2 I]^{-1}{\bf y} $$ and $$ \Sigma_* = k_{**} - k_*[k + \sigma_n^2 I]^{-1} k_* $$.

And that’s it. We can now get our estimate as $$ mu_* $$ and our uncertainty as $$ \Sigma_* $$. So, essentially Gaussian Process regression is just conditioning property of multivariate Gaussians. Of course, we can also incorporate our prior knowledge of the data by specifying the mean function $$ m(\cdot) $$ and the covariance function $$ k(\cdot, \cdot) $$. Below are some figures where I play around with Gaussian Process Regression using different types of observations and different kernels.


<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gpr_x_k2.png" width="600">
</p>
_Figure 8_: This plot shows observations corresponding to a noisy linear model $$f(x) = x$$. Since the data look sort-of-linear, it’s reasonable to first try linear kernel. And voila! Also, this seems to be very Bayesian linear regression and Gaussian Process Regression. With the linear kernel, GPR is just BLR. So, in a sense, GPR is a more general version of BLR.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gpr_xsinx.png" width="600">
</p>
_Figure 9_: In this plot, the underlying function is $$ f(x) = x \mathrm{sin} (x) $$. Obviously a bit more tricky than a linear model. Here, I tried a couple different kernels, the squared exponential and the symmetric kernel. With little hyper-parameter optimization, I was able to get a decent fit for both kernels. But, if I’m just comparing to prediction (blue line) to the underlying function (red line), I think that the symmetric kernel has a nicer fit. Which makes sense, since $$ x \mathrm{sin} x $$ is symmetric. Here is a good opportunity to use our prior to help our kernel choice, even though it wasn’t too much better than the squared exponential kernel.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gpr_sinx.png" width="600">
</p>
_Figure 10_: Here the underlying function is $$ \mathrm{sin}(x) $$. I tried both the squared exponential and the periodic kernel for this. It’s interesting that beyond the range of observations the periodic kernel was able to follow $$ \mathrm{sin}(x) $$ much better. This makes sense, because the squared exponential kernel has no reason to continue with the periodic pattern beyond the range of observations. This is where our prior knowledge would be very helpful in choosing the right kernel. However, in most problems, I think it’d be rare to have to make predictions that much beyond the range of observations, so for the sake of most classification problems, the squared exponential kernel seems to work just fine.

# Conclusion
Gaussian Process regression is powerful general tool for regression problems. And hopefully you learned a bit more about it in this post. But the truth is, this is only the beginning. I know I can still learn a lot more about the theory of kernel functions, ways to make this algorithm faster, and using Gaussian Processes for classification. It’s long road ahead, but everyday we can take one more step in the right direction.

### References
* Carl E. Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006. Online: http://www.gaussianprocess.org/gpml/


