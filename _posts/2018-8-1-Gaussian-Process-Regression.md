---
layout: post
title: Gaussian Process Regresssion
data: 2018-08-01 12:00:00 
category: 
tags:
---

Such a big sounding name! 

Gaussian process regression (GPR) is general method for predicting continuous valued outputs. It's a pretty powerful regression method, stronger than normal linear regression. Some of its more well known applications are in Brownian motion, which describes the motion of particle in a fluid, and in geostatistics, where we're given an incomplete set of 2D points that's supposed to span some space and GPR is used to estimate the gaps between our observations to fill in the rest of the terrain. 

Some prerequisite concepts you should know are the multivariate Gaussian distribution and Bayesian linear regression. I’ll briefly talk about them right now, but I can’t substitute a more thorough reading of these topics.

The main mathematical structure behind GPR is the **multivariate Gaussian distribution**. The important properties of multivariate Gaussians to know for GPR is the _conditioning property_ and the _additive property_. If we write our random vector $$ x \sim \mathcal{N}( \mu, \Sigma ) \in {\rm I\!R}^{n} $$ as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ x = \begin{bmatrix} x_a \\ x_b \end{bmatrix} $$ where $$ x_a = \begin{bmatrix} x_1 \\ . \\ . \\ x_k \end{bmatrix} $$ and $$ x_b = \begin{bmatrix} x_{k+1} \\ . \\ . \\ x_n \end{bmatrix} $$, then

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  $$ \mu = \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}$$ and $$\Sigma =  \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix} $$.

The conditioning property says that the distribution of $$x_a$$ given (conditional on) $$x_b$$ is also multivariate Gaussian! It's distribution is given by

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$x_a | x_b \sim \mathcal{N}( \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b),$$ $$\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$$,

The additive property says that if $$ x \sim \mathcal{N}( \mu_x, \Sigma_x ) $$ and $$ y \sim \mathcal{N}( \mu_y, \Sigma_y ), $$ then 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ x + y \sim \mathcal{N}( \mu_x + \mu_y, \Sigma_x + \Sigma_y ) $$

These are a very neat results! So just remember this for now or come back to it if you forget.

<br>

The next concept that's useful is **Bayesian linear regression**. Without getting too involved in the details of Bayesian linear regression, some important points about BLR is that you can _use your prior knowledge_ of the dataset to help your predictions and you _get a posterior distribution of the prediction_! This is in contrast to other methods which return a single value with no account of the possible variance in that value (these other methods use _Maximum Likelihood estimation_). Bayesian linear regression uses _Maximum a Posteriori estimation_ for the full distribution of the output, called the posterior distribution. Figure 1 shows the difference between these two methods.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/regression1.png" width="600">
</p>
_Figure 1_: Bayesian and Classical (Frequentist) predictions on linear observations. Both are very similar, but one advantage of the Bayesian approach is that it gives the distribution of our prediction. In the figure, the shaded region represents 2 standard deviations above and below the prediction.

<br>

# Gaussian Processes
Just before we get to Gaussian Process regression, it's obviously important to understand Gaussian processes (GPs).

**Gaussian processes** are formally defined a set of random variables $$ \{ f(x) : x \in X \} $$, indexed by elements $$ x $$ (normally time or space) from some index set $$ X $$, such that any finite subset of this set $$ \{ f(x_1),...,f(x_n) \} $$ is multivariate Gaussian distributed.

In theory, these sets can be infinite in size since the index variable (time or space) can go on infinitely and therefore we can think of Gaussian Processes as infinite dimensional extensions of the multivariate Gaussian distribution. But, in practice, we'll always deal with finite sized sets and we can treat them just as we would multivariate Gaussians:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ f( x ) \sim \mathcal{N} ( 0, k( x, x ) ) $$ 

Thinking of Gaussian Processes as infinite dimensional extensions of multivariate Gaussians allows us see them as distributions over random functions. This distribution is specified by a mean function $$ m( \cdot ) $$ and a covariance function $$ k( \cdot, \cdot ) .$$ So another way to denote $$ f $$ is as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ f( \cdot ) \sim \mathcal{GP} ( m( \cdot ), k( \cdot, \cdot ) ) $$ 

Just for preciseness, $$ m( \cdot ) $$ must be a real function and $$ k( \cdot, \cdot ) $$ must be a valid kernel function to have a Gaussian Process.

In the same way that we can sample a random vector from a multivariate Normal distribution, we can sample a random function from a Gaussian distribution. Note the difference here: a vector is finite-sized in how we treat them, but a function is a mapping with a potentially infinite size codomain. And just like the types of vectors we get from a multivariate Normal distribution are determined by its mean vector and covariance matrix, the types of functions we get from a Gaussian Process are determined by the mean function $$ m( \cdot ) $$ and the covariance function $$ k( \cdot, \cdot ) .$$

In determining the shape of the function we sample, the covariance function $$ k( \cdot, \cdot ) $$ is much more interesting than the mean function $$ m( \cdot ) .$$ In this post, I’ll only look at Gaussian Processes with a zero mean function, i.e. $$ m(\cdot) = 0,$$ and you should be able to see the effect this has on the functions we get from the examples below and how varying $$ m(\cdot) $$ would affect the types of functions sampled. I'll show you pictures of functions sampled from 5 different kernels, and in each of the figures, I show a picture with one function sampled from the Gaussian Process and another showing twenty functions sampled from the GP. This isn't meant to be a thorough introduction to kernel function theory, but it'll hopefully give you a better understanding of the different types of kernel out there.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_number.png" width="600">
</p>
_Figure 3_: Functions sampled from a Gaussian Process with a constant kernel $$ k(x_i, x_j) = \sigma^2 $$. In this case, the functions we sample are just constant numbers, but notice how the numbers center around 0 and vary in both directions. This is just like sampling a number from Gaussian distribution $$ x \sim \mathcal{N} (0, 1) $$, except here the numbers are functions! So we see here how GPs can be reduced to simple univariate Gaussian distributions.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_line.png" width="600">
</p>
_Figure 4_: The functions sampled from a Gaussian Process with a linear kernel $$ k(x_i, x_j) = \sigma^2 x_i \cdot x_j $$ are just linear functions. Notice, in the picture on the right, that at each point $$ x $$ the values $$ f(x) $$ center at 0 and vary in proportion to $$ x. $$ That is, $$ f(x) $$ at $$ x = 4 $$ varies much more than at $$ x = 1. $$ It turns out that the distribution of $$ f(x) $$ is distributed at $$ x $$ according to $$ x \sim \mathcal{N} (0, x). $$

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_se.png" width="600">
</p>
_Figure 5_: The functions sampled from a Gaussian Process with a squared exponential kernel $$ k(x_i, x_j) = \sigma^2 \mathrm{exp}( -\dfrac{1}{2 l^2}| x_i - x_j |^2 ) $$ are very smooth functions. These functions are supposed to be infinitely differential at every point. In this figure, the characteristic length scale $$ l $$ is set to 1, but changing $$ l $$ would result in either smoother (with higher $$ l )$$ or more volative (with smaller $$ l $$) functions.

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_symmetric.png" width="600">
</p>
_Figure 6_: The functions sampled from a Gaussian Process with the kernel $$ k(x_i, x_j) = \mathrm{exp} ( - \alpha ( \mathrm{min}( |x_i - x_j|, |x_i + x_j| ) )^2) $$ are symmetric functions about $$ t = 0. $$

<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gp_periodic.png" width="600">
</p>
_Figure 7_: The functions sampled from a Gaussian Process with the kernel $$ k(x_i, x_j) = \sigma^2 \mathrm{exp} ( - \dfrac{2}{l^2} \mathrm{sin}^2 ( \alpha \pi (x_i - x_j) ) ) $$  are periodic functions.

<br>

Each of these kernels has different hyperparameters associated with them that can effect the types of functions sampled. For example, in the periodic kernel $$ k = \mathrm{exp} ( - \mathrm{sin}^2 ( \alpha \pi (x_i - x_j) ) ) $$ in figure 7, making $$ \alpha $$ larger would give functions with much higher frequencies of oscillation, while smaller $$ \alpha $$ would give functions with lower frequencies. This may not be totally relevant, but it's nice to know.

<br>

# Gaussian Process Regression
Finally, we arrive at Gaussian Process Regression. In these types of problems we are given a dataset $$ \{({\bf x}_{i} , y_{i}) |i=1,...,m\} $$, where $$ y = f({\bf x}) + \epsilon $$ is a noisy observation of the underlying function $$ f({\bf x}) $$. It is important to assume that the noise is $$ \epsilon \sim \mathcal{N}(0,\sigma^2) $$.

The goal is to predict the $$y$$ values for future $$x$$’s. If we knew what the function $$ f({\bf x}) $$ was then we wouldn’t need to do GP regression because we could just plug in our future ’s and get a perfect prediction. Unfortunately, we don’t know $$ f({\bf x}) $$. So, what we’ll do is sample some functions from a Gaussian Process and go from there.

In theory, we can sample an infinite number of functions and choose only the ones that fit our data. But, in practice this is obviously not feasible. So, if we write down our model again

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ y = f({\bf x}) + \epsilon $$ where $$ f( {\bf x}) \sim \mathcal{GP}(0, k(\cdot, \cdot)) $$ and $$ \epsilon \sim \mathcal{N}(0,\sigma^2), $$

then we’ll notice that because $$ f({\bf x}) $$ is multivariate Gaussian distributed (from the definition of Gaussian Processes) and $$ \epsilon $$ is Gaussian distributed (by assumption), then $$y$$ must also be multivariate Gaussian distributed, i.e. $$ y \sim \mathcal{N}(0, k(\dot{},\dot{}) + \sigma^2)! $$

If we have our labeled dataset $$ \{({\bf x}_{i} , y_{i}) |i=1,...,m\} $$ and we also have the unlabeled dataset $$ \{ {\bf x}_{i} |i=m+1,...,n\} $$
for which we want to predict , then a reasonable assumption is that both  and  come from the same distribution, namely

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ \begin{bmatrix} y \\ y_* \end{bmatrix} \sim \mathcal{N}( 0, \begin{bmatrix} k(x,x) + \sigma^2I && k(x,x_*) \\ k(x_*, x) && k(x_*, x_*) + \sigma^2 I \end{bmatrix} ) $$.

where $$ x $$ denotes labeled data and $$ x_* $$ denotes unlabeled data. So, from the conditioning property of multivariate Gaussians, we can get our conditional distribution of $$ y_* $$ given $$ x $$ as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ {\bf y}_*|X, {\bf y}, X_* \sim \mathcal{N}( \mu_*, \Sigma_*) $$

where

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ \mu_* = k_*[k + \sigma_n^2 I]^{-1}{\bf y} $$ and $$ \Sigma_* = k_{**} - k_*[k + \sigma_n^2 I]^{-1} k_*. $$

And that’s it. We can now get our estimate as $$ \mu_* $$ and our covariance as $$ \Sigma_* $$. So, essentially Gaussian Process regression is just conditioning property of multivariate Gaussians. Of course, we can also incorporate our prior knowledge of the data by specifying the mean function $$ m(\cdot) $$ and the covariance function $$ k(\cdot, \cdot) $$. 

Below is the segment of code that’s calculates the estimate $$( \mu_*)$$ and the covariance $$ ( \Sigma_* ). $$ The algorithm I use is taken from Rasmussen et al, chapter 2. Instead of directly taking the inverse of the prediction kernel matrix, they calculate the Cholesky decomposition (i.e. the square root of the matrix), which takes $$ O(n^3) $$ time.

```python
n = len(X)
K = variance( X, k )  // returns variance of X defined by kernel function k
L = np.linalg.cholesky( K + noise*np.identity(n) )    
    
# predictive mean
alpha = np.linalg.solve( np.transpose(L), np.linalg.solve( L, Y ) )
k_hat = covariance( X, x_hat, k ) // returns covariance of X and x_hat defined by kernel function k
mu_hat = np.matmul( np.transpose(k_hat), alpha )
    
# predictive variance
v = np.linalg.solve( L, k_hat )
k_hathat = variance( x_hat, k )  
a = np.matmul( np.transpose(v), v )
covar_hat = k_hathat - np.matmul( np.transpose(v), v )
var_hat = covar_hat.diagonal()
```

<br>

Just for you to see an example of Gaussian process regression (with a squared exponential kernel) in work, Figure 8 shows the evolution of the posterior distribution as more observations are made. Before any observations, the mean prediction is zero and shaded area is 2 standard deviations from the mean (1.96 in this case). After the first observation is made, prediction changes slightly and the variance shrinks near the region at that point. Subsequent observations produce better predictions and smaller uncertainties. After ten observations are made, we can already see a pretty nice curve and prediction. 

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/evolution.png" width="1000">
</p>
_Figure 8_: These pictures shows how the posterior distribution of the prediction changes as more observations are made. The GP here uses the squared exponential kernel.

<br>

Below are some figures where I play around with Gaussian Process Regression with different types of observations and different kernels. Figure 9 (below) starts it off with linear observations predicted using a GP with a linear kernel. This can also be seen as just doing Bayesian linear regression and shows how Gaussian Process Regression is a more general version of Bayesian linear regression.  

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gpr_x_k2.png" width="600">
</p>
_Figure 9_: This plot shows GP regression with a linear kernel o observations corresponding to a noisy $$f(x) = x.$$

<br>

In Figure 10 (below), I try to model noisy observations from the nonlinear function $$ f(x) = x \mathrm{sin} (x). $$ Obviously this is a bit more trick than a basic linear function, so I try a couple different kernels for this: the squared exponential kernel and the symmetric kernel. In my opinion, the difference in performance between the two isn’t really that impressive, but the variance from the symmetric kernel does seem to be little nicer looking that from the SE kernel. Although, I should warn you that I didn’t optimize any of the hyperparameters in this post, which would obviously affect the results.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gpr_xsinx.png" width="600">
</p>
_Figure 10_: Gaussian processes regression using the squared exponential kernel and the symmetric kernel. The observations are based on the function $$ f(x) = x \mathrm{sin} (x). $$

<br>

In Figure 11 (below), I use the squared exponential kernel and the periodic kernel to model noisy observations of $$ \mathrm{sin}(x). $$ It’s interesting that beyond the range of observations $$(-5, +5)$$ the periodic kernel is able to follow $$ \mathrm{sin}(x) $$ much better. This makes sense, because the squared exponential kernel has no reason to continue with the periodic pattern beyond this range. This is prior knowledge about the dataset would be very helpful in choosing the proper kernel. However, in most problems, I think it’d be rare to have to make predictions far beyond the range of observations, so for the sake of most problems, the squared exponential kernel seems to work just fine.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/gaussian-process-regression/gpr_sinx.png" width="600">
</p>
_Figure 11_: Gaussian processes regression using the squared exponential kernel and the periodic kernel. The observations are based on the function $$ \mathrm{sin}(x) .$$

<br>

# Conclusion
Gaussian Process regression is powerful all-purpose, general tool for regression problems. Unfortunately, one of the reasons it not used as widely is because of the time complexity of the algorithm, which is $$O(n^3)$$ time from taking the inverse of a large matrix. 

Hopefully, though, you learned a bit more about Gaussian Process Regression in this post. And one exciting thing is that this is only the beginning. I know I can still learn a lot more about the theory of kernel functions and building my own kernels, working with higher dimensional GPs, and applying Gaussian Processes for classification. It’s long road ahead.

### References
* Carl E. Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006. Online: [http://www.gaussianprocess.org/gpml/](http://www.gaussianprocess.org/gpml/)


