---
layout: post
title: Learning Physics from the Machine
data: 2018-07-01 12:00:00 
category: 
tags:
---
50 first

<div align="justify">
In particle physics, people smash particles together at very high speeds to see what happens. The higher the speed, the more likely we can see something we’ve never seen before; a new and exciting discovery. For example, the Higgs Boson, which holds the secret to the nature of mass, was discovered in 2013 at the Large Hadron Collider (LHC). These types of experiments can help tell how us what are the most fundamental components of all matter.

<br><br>

</div>

[<img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig1.png" 
       width="800"
       class="center">](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/CONFNOTES/ATLAS-CONF-2012-161/)
<div align="center">
       <b>Fig. 1</b>: ATLAS event display of two b-jets and two electrons.
</div>

<br>

<div align="justify">
However, seeing new and exciting particles are rare. So we have to look at a lot of events (collisions) in the hope that we might see something cool. This is why experiments like the LHC produce approximately 600 million events per second, which is then filtered to around 100-200 events/second for analysis. After some time, that’s a lot of data, which can be hard to analyze. To classify these events, physicists have come up with their own algorithms to produce high-level engineered variables from the images.

<br><br>

This has led to the application of machine learning in particle physics, specifically neural networks. Neural networks deal with learning patterns from data; more data means better ability to see patterns and make discoveries. In recent work by 
<a href="https://arxiv.org/abs/1603.09349">Baldi, et al. (2016)</a>, neural nets have actually outperformed high level engineered variables for tasks such as classification.

<br><br>
</div>

<div style="text-align:center">
  <figure class="sreenshot">
    <a href="https://arxiv.org/abs/1603.09349">
      <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig2.png"
        width="400">
    </a>
  </figure>
</div>
<div align="justify" width="70%">
  <b>Fig. 2</b>: Signal efficiency versus background rejection (inverse of efficiency) for various classifiers such as deep neural networks trained on collision images, boosted decision trees trained on expert variables, jet mass individually, and expert variables (in conjunction with mass) designed by physicists. Clearly the machine learning techniques outperform the engineered variables.
</div>

<br>

<div align="justify">
However, neural networks are tricky. They work well, but there are many open questions about them. Why do they work so well? What is the neural net learning from the data? Why is it learning that particular thing?

<br><br>

The objective of this work is to interpret the information that the neural network has learned in terms that physicists could understand. The motivation for this is to gain insight and possibly learn something new about the nature of the physics involved.
</div>

# Translating from machine to human
<div align="justify">
This is the tricky part. Taking the neural network output and translating it into something physicists can understand. 

<br><br>

For this particular work, we looked at the problem of classifying jets of particles using substructure information. So the neural network would take an image of the jet substructure and classify it as either signal or background. Specifically, neural networks make these decisions using a set of numbers (weights) that it learns and non-linear transformations. Thus far, physicists can not make sense out of these weights; in fact, no one, not just physicists, can really understand these weights. They just happen to work and that’s that. 

<br><br>

So, we want to take the information from these weights and translate it into something physicists can understand. Based on certain arguments and assumptions in quantum physics involving particle collisions which are beyond my scope, physicists can understand these jet images based on functions involving the momentum fraction in each pixel in the jet. The first three terms in this family of functions is expressed as:

<br>
</div>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/functions.png"
       width="500">
</p>

where the momentum fraction $$z_i$$ of pixel $$i$$ is:
<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fractional_transverse_momentum.png"
       width="150">
</p>

<div align="justify">
So essentially we want to take the information from the weights of the neural network and test its similarity to these functions. If they are similar, then perhaps the neural network is learning something related to the theory of these functions.

<br><br>

Now comes the question of how to test similarity. We need a metric that reflects the task of using the information from the image for classification. However, it must also be invariant to non-linear 1-to-1 transformations of the two functions, because of the non-linear nature of neural networks. So, we consider the decision surfaces defined by the threshold on the function output. By decision surface, we mean the hyperplane which separates what the function believes to be signal and background. Specifically, we consider two functions similar if, for any pair of points, they produce the same signal-to-background ordering. This is expressed through what we call the discriminant ordering between two functions, f and g, for a pair of points:

</div>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/DO-pair.png"
       width="500">
</p>

For the discriminant ordering over the entire sample, we use integration:

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/total-DO.png"
       width="500">
</p>

<div align="justify">
After normalizing, this integral evaluates to zero if the decision surfaces of the two functions have no similarity and one if they have identical decision surfaces. 
	
<br><br>

As an example, below is table that shows between the discriminant ordering between pairs of the physicist engineered jet substructure variables:
</div>
