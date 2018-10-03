---
layout: post
title: Learning Physics from the Machine
data: 2018-07-01 12:00:00 
category: 
tags:
---
1 2

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
       width="480">
</p>

<div align="justify">
After normalizing, this integral evaluates to zero if the decision surfaces of the two functions have no similarity and one if they have identical decision surfaces. 
	
<br><br>

As an example, below is table that shows between the discriminant ordering between pairs of the physicist engineered jet substructure variables:
</div>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/DO-high-level-variables.png"
       width="500">
</p>
<div align="justify" width="70%">
  <b>Fig. 3</b>: Similarity of the decision surfaces between the six physicist engineered variables, as measured the discriminant ordering metric defined in the text above. A zero indicates no similarity, a one indicates perfect similarity.
</div>

# Application: Reduce and orthogonalize
<div align="justify">
Now there are a few different approaches our team took to generate insight regarding this problem. The first attempts to explain the gap between neural networks trained on the images and the performance of the high level engineered variables by searching for a new high level engineered variable that captures the lost information. The second attempts to gain insight into the neural network strategy on the images by mapping it into our space of functions. In the third, we look at the information content in the high level engineered variables by attempting to reduce and orthogonalized them into a minimal set. Since my work involved the last approach, this post will explore the that more deeply. (For more information on the first two approaches, see the paper.)

<br><br>

My approach was take the set of high level engineered variables and reduce them to as small a set as necessary while maintaining the original classification power as well as being orthogonal from each other. By orthogonal, I mean that they don’t contain the same information for classification purposes, that is, they have a small discriminant ordering between each other.

<br><br>

The procedure attempts to iteratively replace each of the high level engineered variables with a learned variable that captures the same classification power but minimizes the information overlap with the other variables. We first scan the list variables, selecting the one which has the largest negative impact on the classification power when it is removed from the set. This variable is then replaced by a neural network which takes as input the jet image and returns as output a single number, to be later identified as a potential new high level variable. In order to maximize the independence of the new subnet output from the existing variables, the networks are trained together with an adversarial network, which uses the subnet output to attempt to recover the values of the other HL variables. Figure 3 depicts the structure of this first step. In the next iteration, the subnet is frozen and a second HL variable is selected for replace by a new subnet and orthogonalization process. This procedure is repeated until removing HL variables has no impact.
</div>

<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig4.png"
       width="700">
</p>
<div align="justify">
  <b>Fig. 4</b>: Step 1 analyzes the output of a subnet with only jet image features into a single output and trained as part of a larger classification network which also has access to high level engineered features except for one feature which is being replaced; the subnet is also trained by an adversarial network, which penalizes it if the subnet output can be used to predict the other five high level features. The subnet output is also compared the function space of possible strategies using the discriminant ordering metric. Step 2 analyzes the output of a second subnet, trained as shown to replace a second high level feature with an orthogonalized version.
</div>

<br>


<div align="justify">
We found that of the six high level variables, $$M_{jet}$$, $$C_2^{\beta=2}$$, and $$C_2^{\beta=1}$$ (in this order) were best able to be transformed so as to capture the same classification information from the original six high level variables while being orthogonal to other variables. We used ROC curve AUC as the metric for the variables’ ability for classification ability and discriminant ordering as the metric for orthogonality. 
</div>

<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/do_orthogonal.png"
       width="500">
</p>
<div align="justify">
  <b>Fig. 5</b>: Comparison of the similarity of classification by original HL variable and orthogonalized versions.
</div>

<br>

<div align="justify">
We also compared the 3 orthogonalized variables to physics functional space defined by two parameters.
</div>

<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig6.png"
       width="1200">
</p>
<div align="justify">
  <b>Fig. 6</b>: Comparison of the three orthogonalized HL variables with points in the functional space defined by two parameters.
</div>

# Conclusion
Power of neural networks and usefulness.
Future work: exploring larger part of function space, exploring hidden nodes.

Paper currently being prepared for submission.
