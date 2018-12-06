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

An interesting development in machine learning in the past few years has been an interest in deep reinforcement learning, which is just reinforcement learning with deep neural networks. Originally, I was going to introduce and explain just reinforcement learning, talk about agents and policies and value functions, etc. But there are already some really good resources out there that introduce reinforcement learning and I wanted to try something different (link to resources). So, in this post I’ll talk about a paper that applies reinforcement learning to computer vision!

The paper I talk about is *Recurrent Models of Visual Attention* by Mnih, Heess, Graves, and Kavukcuoglu (link to paper). This paper introduced a method for image classification that uses a sequence of “glimpses” on different regions of the image to predict the class. The authors took their inspiration from human perception, which focuses on selective regions instead of processing the entire scene at once (as a convolutional neural net would). This means that less computational resources (model parameters) are used since our model doesn’t have to process the entire image. Also, it means that the model can potentially ignore clutter and other irrelevant parts of the image for classification. 

## Model & Method
<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/model.png" width="600">
</p>
_Figure 1_: 
**A) Glimpse sensor** that takes an image $$x$$ and a location $$l_{t-1}$$ and returns glimpses $$\rho$$ of image at the given location. 
**B) Glimpse network**: takes an image $$x$$ and a location $$l_{t-1}$$, receives the glimpses $$\rho$$ from the glimpse sensor, and then processes $$\rho$$ and the $$l_{t-1}$$ through fully connected layers to get a representation vector of the glimpses $$g_t$$.
**C) Model**: An Recurrent Neural Network (RNN) model that takes an image $$x$$ and a location $$l_{t-1}$$ (initially chosen at random) in the glimpse representations $$g_t$$ from the glimpse network and uses it to update the RNNs internal state $$h_{t-1}$$ to get $$h_{t}$$. The internal state $$h_t$$ is then used by the location network to determine the next location $$l_{t}$$, which is then fed back into the RNN. This sequence is repeated for a fixed number of times until the last glimpse/action is reached, where the internal state $$h_t$$ is fed into the action network, which predicts the image class. 

<br>

#### Architecture of the model
Figure 1 shows the basic framework of the RAM (Recurrent Attention Model, what they call it) they used, which is composed of 5 smaller neural networks:
* Glimpse sensor
* Glimpse network
* Core Recurrent Neural Network (RNN) network
* Action network
* Location network

Essentially, they use an RNN network that takes in an image, sequentially gathers information about the image using glimpses from different parts of the image, and then classifies the image. 

#### Training Procedure
The objective in this problem is to maximize the expected total reward, i.e. find $$ \theta^* = {\rm arg max}_\theta E_{p(s_{1:T}; \theta) ( \Sigma_{t=1}^T r_t ) $$. Here $$s_{1:t} = x, l_1, a_1, …, x_t, l_t, a_t$$ is the distribution of possible interaction sequences. 

Apparently, this turns out to be quite difficult. Choosing the sequence of locations to glimpse is not some function that we can backprop on (if it is, it is high dimensional and quite complex and may change over time). So, Mnih et. al. used reinforcement learning to train a policy $$\pi$$ to choose actions given interactions. The policy in this case is defined by the RNN above: $$\pi ((l_t, a_t) | s_{1:t}; \theta)$$. They trained to policy using the policy gradient algorithm (aka REINFORCE), which is just gradient ascent on the policy parameters. 

$$\nabla_\theta J(\theta) = \Sigma^T_{t=1} E_{p(s_{1:T};\theta)} ( \nabla_\theta \log \pi ( a_t | s_{1:T} ; \theta ) (R - b_t) ) $$

$$ \ \ \ \ \ \ \ \approx \dfrac{1}{M} \Sigma^M_{i=1} \Sigma^T_{t=1} ( \nabla_\theta \log \pi ( a^i_t | s^i_{1:T} ; \theta ) (R^i - b_t) ) $$

One note: the $$b_t$$ term is the baseline reward. It’s generally added in policy gradient to reduce the variance of the gradient, which helps in training policies. 

For the classification problem (last action), they  used a cross entropy loss function and backpropped the loss through the differentiable parts of the network. The location network (which decided where the next glimpse should be) was trained via policy gradients. 

My descriptions in this section are brief overviews of the entire model and method. See the paper if you’re interested in other details such as learning rates, the number of hidden units, etc.

<br>

## My Results
My goal was to reproduce the results from this paper. Specifically, I wanted to reproduce the baseline and best results with the MNIST dataset and the cluttered translated MNIST dataset. This involved writing the code that created the network and training the network. My code is available on my GitHub and I trained it on a GPU from Google Colab. 

#### MNIST results - Baseline (FC & Conv Net) and RAM 
<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/resA.png" width="600">
</p>

#### Cluttered Translated MNIST - - Baseline (FC & Conv Net) and RAM
The cluttered translated MNIST dataset is a customized dataset where an original 28 by 28 MNIST image is padded to size 60x60, the translated such that the digit is placed at a random location, and then cluttered by adding 8 by 8 random sub patches from other random MNIST digits to random locations of the image. Example cluttered translated MNIST images can be seen in figure 2.

<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/cluttered_translated_MNIST.png" width="600">
</p>
_Figure 2_: Pictures and labels of the 60 by 60 cluttered translated MNIST images as described in the original paper. I generated these images by first placing an MNIST digit in a random location on a 60 by 60 blank image and then adding random 8 by 8  sub patches from other random MNIST digits to random locations of the image.

<br>
<p align="center">
    <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/recurrent_attention_model/resB.png" width="600">
</p>
<br>

##### Thoughts on performance
Although the performance of my networks were pretty close to those of the authors, I admit I was a bit disappointed that my results didn’t exactly quite match theirs. I believe I was faithful to the procedures and design described in the paper and trained my networks for quite a while (i.e. until the validation accuracy didn’t improve for 25 epochs). 

Some reasons for my performance that I can think of that include a lack of an decent parameter search for the learning rate and location network standard deviation. In training, I only tried a couple of different values out, which is obviously not ideal. Upon reflection, I think it’s fair to assume that a small increase an error can be attributed to non-optimal parameters. I’ll probably try and train this network a few more times with different param values and update this post as it goes along.

I'm not surprised that in the plain MNIST, the convolutional net outperformed RAM. This makes sense since it has access to nearly the same information. I say "nearly" and not "more" because the kernel size (10 by 10) and stride length (5) are pretty big whereas the RAM glimpse size (8 by 8) is a bit more fine grained (although not convolutional). This isn't a big deal, but just a small remark.

It’s cool to see the performance of RAM on the cluttered translated MNIST in relation to the baselines. It’s not a huge surprise though considering the types of baselines used - 2 layer conv nets with large kernels and strides. I’m curious to see how a better network, perhaps a pretrained like VGG or ResNet, might perform.

##### Thoughts on number of parameters
Looking at the number of parameters it didn’t seem like there really was a huge difference. In fact, based on the descriptions in the paper, the conv net baseline actually had the least amount of parameters. Of course, a larger baseline would be different, but I only considered what was in the paper. 

<br>

## Concluding Remarks
I choose this project because I’ve recently been watching and following along with the Berkeley deep reinforcement learning lectures online and wanted to see some of the concepts talked about in action. This particular paper drew my interest because it applies reinforcement learning to computer vision, which I do have some experience in. So many applications (it seems) of reinforcement learning that people talk about have to do with games or robots, which, don’t get me wrong, are very interesting. But these areas now seem so saturated. I was curious about different domains where RL can make an impact, and doing this project was perfect for me to see the policy gradient in action for image classification. 

I know I still have a long way to go. This project was not trivial for me. My god, this must have took almost a whole month. But, this was good practice for me and something I enjoyed doing a bit more than the intro type post (like that of Gaussian Process Regression). Although I’m not quite sure exactly what my next little project will be, I’m currently leaning towards doing another paper type project, you know, read an interesting paper and try to replicate the results. So stay tuned!

Also… if there are any typos or mistakes that you find, or if you just wanna contact me for fun, please email me so I can correct it :)
