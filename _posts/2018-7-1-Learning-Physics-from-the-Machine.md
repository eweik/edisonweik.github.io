---
layout: post
title: Learning Physics from the Machine
data: 2018-07-01 12:00:00 
category: 
tags:
---
o no nos

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
<a href="https://arxiv.org/abs/1603.09349">Baldi, et al. (2016)</a>
, neural nets have actually outperformed high level engineered variables for tasks such as classification.

<br><br>
</div>

[<img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig2.png" 
       width="400"
       aligh="center](https://arxiv.org/abs/1603.09349)
<div align="justify" width="70%">
  <b>Fig. 2</b>: Signal efficiency versus background rejection (inverse of efficiency) for various classifiers such as deep neural networks trained on collision images, boosted decision trees trained on expert variables, jet mass individually, and expert variables (in conjunction with mass) designed by physicists. Clearly the machine learning technique outperform the engineered variables.
</div>

<br>

<div align="justify">
However, neural networks are tricky. They work well, but there are many open questions about them. Why do they work so well? What is the neural net learning from the data? Why is it learning that particular thing?

<br><br>

The objective of this work is to interpret the information that the neural network has learned in terms that physicists could understand. The motivation for this is to gain insight and possibly learn something new about the nature of the physics involved.
</div>

# Translating from machine to human
