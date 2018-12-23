---
layout: post
title: Recurrent Attention Models
data: 2018-12-01 12:00:00 
category: 
tags:
---

A look at an application of reinforcement learning in computer vision!

## Intro
Hello world! Since I’m still a beginning blogger, I’m going to try something new for this post. Instead of introducing and explaining a topic, I’m going to talk about a research paper I recently read and whose results I tried to reproduce. Thank you for reading and I hope you find this as compelling as I do…

An interesting development in machine learning in the past few years has been the progress in deep reinforcement learning, i.e. reinforcement learning with deep neural networks. Originally, I was going to introduce and explain just reinforcement learning: talk about agents and policies and value functions and maybe overview basic algorithms like Q-learning or policy iteration. But there are already some really good resources out there that introduce reinforcement learning and I wanted to try something different. So, in this post I’ll talk about a paper that applies reinforcement learning to computer vision!

The paper I talk about is [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by Mnih, Heess, Graves, and Kavukcuoglu. This paper introduced a method for image classification that uses a sequence of “glimpses” on different regions of the image to predict the class. The authors took their inspiration is from human perception. They claim that their model uses less computational resources (model parameters) because it doesn’t have to process the entire image. Also, they state that the model can potentially ignore clutter and other irrelevant parts of the image for classification. In this post, I’ll show my attempt to replicate their results and test their claims.

## Model & Method
[<img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/model.png" 
       width="800"
       class="center">](https://arxiv.org/abs/1406.6247)

_Figure 1_: 
**A) Glimpse sensor** that takes an image $$x$$ and a location $$l_{t-1}$$ and returns glimpses $$\rho$$ of image at the given location. 
**B) Glimpse network**: takes an image $$x$$ and a location $$l_{t-1}$$, receives the glimpses $$\rho$$ from the glimpse sensor, and then processes $$\rho$$ and the $$l_{t-1}$$ through fully connected layers to get a representation vector of the glimpses $$g_t$$.
**C) Model**: An Recurrent Neural Network (RNN) model that takes an image $$x$$ and a location $$l_{t-1}$$ (initially chosen at random) in the glimpse representations $$g_t$$ from the glimpse network and uses it to update the RNNs internal state $$h_{t-1}$$ to get $$h_{t}$$. The internal state $$h_t$$ is then used by the location network to determine the next location $$l_{t}$$, which is then fed back into the RNN. This sequence is repeated for a fixed number of times until the last glimpse/action is reached, where the internal state $$h_t$$ is fed into the action network, which predicts the image class. 

<br>

#### Architecture of the model
Figure 1 shows the basic framework of the RAM (Recurrent Attention Model, what they call it), which is composed of 5 smaller neural networks:
* Glimpse sensor
* Glimpse network
* Core Recurrent Neural Network (RNN) network
* Action network
* Location network
 
Essentially, they use an RNN network that takes in an image, sequentially gathers information about the image using glimpses from different parts of the image, and then classifies the image using the information. 

#### Training Procedure

The objective in this problem is to maximize the expected total reward, i.e. find $$ \theta^* = \text{arg max}_\theta E_{p(s_{1:T} ; \theta)} \lbrack \sum^T_{t=1} r_t \rbrack. $$ Here $$s_{1:t} = x_1, l_1, a_1, …, x_t, l_t, a_t$$ indicates that the distribution is over the possible glimpse sequences. 

Apparently, this turns out to be quite difficult. Choosing the sequence of locations to glimpse is not some function that we can backprop on (if it is, it is high dimensional and quite complex and may change over time). So, Mnih et. al. used reinforcement learning to train a policy $$\pi$$ to choose actions given interactions. The policy in this case is defined by the RNN above: $$\pi ((l_t, a_t) \vert s_{1:t}; \theta)$$. They trained to policy using the policy gradient algorithm (aka REINFORCE), which is just gradient ascent on the policy parameters. 

$$\nabla_\theta J(\theta) = \sum^T_{t=1} E_{p(s_{1:T};\theta)} \lbrack \nabla_\theta \log \pi ( a_t | s_{1:T} ; \theta ) (R - b_t) \rbrack $$

$$ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \approx \dfrac{1}{M} \sum^M_{i=1} \sum^T_{t=1} \lbrack \nabla_\theta \log \pi ( a^i_t | s^i_{1:T} ; \theta ) (R^i - b_t) \rbrack $$

The $$b_t$$ term is the baseline reward. It’s generally added in policy gradient to reduce the variance of the gradient, which helps in training policies. 

For the classification problem (last action), they used a cross entropy loss function and backpropped the loss through the differentiable parts of the network. The location network (which decided where the next glimpse should be) was trained via policy gradients. 

My descriptions in this section are brief overviews of the entire model and method. See the paper if you’re interested in other details such as learning rates, the number of hidden units, etc.

<br>

## My Results
My goal was to reproduce the results from this paper. Specifically, I wanted to reproduce the baseline and best results with the MNIST dataset and the cluttered translated MNIST dataset. This involved writing the code that created the network and training the network. I trained it on a GPU from Google Colab; my code is available on my [GitHub page](https://github.com/eweik).

For each of the datasets, I show a table with the results of the authors and my results and include a gif of the sequence of glimpses for some images.

#### MNIST results - Baseline (FC & Conv Net) and RAM 
The MNIST dataset consists of 28 by 28 pixels of handwritten letters from 0 to 9. 

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/table1.png" width="600">
       <br>
    Table 1
</p>
<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/mnist_glimpses.gif" width="800">
</p>

#### Cluttered Translated MNIST - Baseline (FC & Conv Net) and RAM
The cluttered translated MNIST dataset is a customized dataset where an original 28 by 28 MNIST image is padded to size 60x60, then translated such that the digit is placed at a random location, and finally cluttered by adding 8 by 8 random sub patches from other random MNIST digits to random locations of the image. Example cluttered translated MNIST images can be seen in figure 2.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/cluttered_translated_MNIST.png" width="600">
</p>
_Figure 2_: Pictures and labels of the 60 by 60 cluttered translated MNIST images as described in the original paper. I generated these images by first placing an MNIST digit in a random location on a 60 by 60 blank image and then adding random 8 by 8  sub patches from other random MNIST digits to random locations of the image.

<br>
<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/table2.png" width="600">
       <br>
    Table 2
</p>
<br>

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/mnist_cl_tr_glimpses.gif" width="800">
</p>
For the cluttered translated MNIST RAM gif above, I only show the glimpses from the central/smallest glimpse. In the paper, the authors use 3 different glimpse scales for each glimpse in the sequence (see paper for more details).

<br>

##### Thoughts on performance
Unfortunately, my accuracy did not match Mnih et al’s in the normal MNIST. To me, this makes sense because the RAM model as the author’s describe it seems to consist entirely of fully connected layers, except without the ability to observe the entire image (which the fully connected baseline does). So, at least to me, I wouldn’t necessarily expect it to outperform the fully connected baseline, much less the convolutional baseline.

In the case of the cluttered translated MNIST, it’s nice to see that my networks performed on par with the original authors’. Here, I did expect the RAM model to outperform the baselines because of the extra noise that would throw off the baseline networks. Ideally, the RAM would learn to focus on just the digits and ignore the noise, and the accuracy of the model suggests that it’s able to do something similar to this. 

Back to the original MNIST. Unlike myself, the authors were able to train a RAM to outperform the baselines. I personally believe this may be an artifact of optimization. Training policy gradient is not fun. Tuning the hyperparameters and scheduling the decay of the learning rate were very long processes for this model and the performance was very sensitive these things. Although I did try quite a few different hyperparameters for each model, I was not rigorous. I’m sure if I spent more time trying out different learning rates and location standard deviations, then I could’ve squeezed out perhaps 1% more accuracy. But, in the case of the original MNIST model, I still don’t think it would’ve outperformed a good baseline.


##### Thoughts on number of parameters

<br>
<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/table3.png" width="600">
       <br>
    Table 3
</p>
<br>

Looking at the number of parameters it didn’t seem like there really was a huge difference. In fact, based on my implementations, the conv net baseline actually had the least amount of parameters. 

However, I get the sentiment the authors are giving. While the RAM may have had more parameters than the conv baseline, it also scaled much better with the input size than the conv baseline did. 

<br>

## Concluding Remarks
I choose this project because I’ve recently finished watching the Berkeley deep reinforcement learning lectures online and wanted to put my hand at trying some of the concepts talked. This particular paper drew my interest because it applies reinforcement learning to computer vision, which I do have some experience in. It seems that so many applications of RL that people talk about have to do with games or robots. Don’t get me wrong, are very interesting, but these areas now seem so saturated. I was curious about different domains where RL can make an impact, and doing this project was perfect for me to see the policy gradient in action for image classification. 

I know I still have a long way to go. This project was not trivial for me. My god, this must have took almost a whole month. But, this was good practice for me and something I enjoyed doing a bit more than the intro type post (like that of Gaussian Process Regression). Although I’m not quite sure exactly what my next little project will be, I’m currently leaning towards doing another paper type project, you know, read an interesting paper and try to replicate the results. So stay tuned!

Also… if there are any typos or mistakes that you find, or if you just wanna contact me for fun, please email me so I can correct it :)
