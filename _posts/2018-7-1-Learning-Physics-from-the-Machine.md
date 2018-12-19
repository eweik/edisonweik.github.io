---
layout: post
title: Learning Physics from the Machine
data: 2018-07-01 12:00:00 
category: 
tags: 
- research
- physics
---
<div align="justify">
Hello world! 
	
<br><br>

Welcome to my little blog, where I will write about general things in science that interest me. Sometimes I'll talk about my personal work, sometimes I'll try and explain a concept I find obscure, sometimes I'll talk about random stuff. It's my blog, I can do with it what I want (whatever that is). 

<br><br>

In this first post I'll talk about a project I worked on as an undergrad at UCI with UCI ATLAS team. It involves some physics and some machine learning, but hopefully I'm not too technical in my descriptions. I'll talk a little bit about the background and then dive into method and results. Bear with me now, I'm only just beginning so there probably be some mistakes some wear, but hopefully not too many. Okay, let's get to it...

<br><br>

A lot of particle physics research essentially involves people smashing particles together at very high speeds to see what happens. The more energy a collision has, the more likely we can see something we’ve never seen before. For example, the Higgs Boson, a very important particle for understanding mass, was discovered in 2013 at the Large Hadron Collider (LHC). These are the types of experiments that help us fathom the most basic components of all matter.
</div>

<br>


[<img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig1.png" 
       width="800"
       class="center">](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/CONFNOTES/ATLAS-CONF-2012-161/)
<div align="center">
       <b>Fig. 1</b>: ATLAS event display of two b-jets and two electrons.
</div>

<br>

<div align="justify">
These experiments, however, produce a lot of data in order to increase the likelihood of seeing something new. The LHC produces approximately 600 million events per second (which is then filtered to around 100-200 events/second for analysis) and has millions of sensors! That's a lot of data at very high dimensions. 

<br><br>

To classify these events, physicists have come up with their own algorithms that take in the raw data and produce “high-level engineered variables.” That is, they use their own hueristics for a custom dimensionality reduction algorithm. In this post, the high level variables we look at are:

</div>

<p><p> $$ M_{jet}, \ C_2^{\beta=1}, \ C_2^{\beta=2}, \ D_2^{\beta=1}, \ D_2^{\beta=2}, \ \tau_{21}^{\beta=1} $$ </p></p>

<div align="justify">
These high level variables should capture the same information as the raw data (low level variables).
But, recent work has shown this to not necessarily be the case. <a href="https://arxiv.org/abs/1603.09349">Baldi, et al. (2016)</a> showed that deep neural networks (DNNs) trained on raw data outperformed DNNs trained on high level variables in classification (figure 2). So, maybe physics heuristics aren't optimal. New questions start to pop up: What aren't physicists accounting for? What is the neural network doing differently?
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
These are tricky questions to answer. Neural networks work well, but there are still some mysteries about them. Why do they work so well? What structure are they learning? Why that structure?

<br><br>

The objective of the work which this post is based on is to interpret what the neural network has learned in physics terms. The motivation is that maybe physicists can gain insight and possibly learn from neural nets something new about the nature of the physics of high energy collisions.
</div>

# Translating from machine to human
<div align="justify">
The first part of this project involves translating the output of a neural net (which is just a linear mapping from the last layer) into something physicists can understand. In this particular work, we looked at the problem of classifying jets/streams of particles using substructure information in pixel images.

<br><br>

Based on certain arguments and assumptions in quantum physics which are frankly above my head, physicists can understand these jet images through functions involving the momentum fraction in each pixel of the image. The first three terms in this family of functions is expressed as:

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
So we want to take the output from the neural net and compare it to these functions. If they are similar, this suggests that the neural network is learning something related to the physics of these functions. Physicists could then explore these functions and possibly learn something new.

<br><br>

To compare these values, we needed a metric that reflected the task of using the information from the low level data (image) for classification, but, was also invariant to non-linear 1-to-1 transformations, because of the non-linearity of neural networks. So, we considered the decision surfaces of the function output, that is, the hyperplane which separates what the function believes to be signal and background. The space where the hyperplane is defined would depend on the function or neural net at hand; but the important part is the threshold for which is separates signal and background. Specifically, we consider two functions similar if, for any pair of points, they produce the same signal-to-background ordering. This is expressed through what we call the <i>discriminant ordering</i> between two functions, <i>f</i> and <i>g</i>, for a pair of points <i>x</i> and <i>x'</i>:

</div>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/DO-pair.png"
       width="500">
</p>

For the discriminant ordering over the entire sample, integrate:

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
  <b>Fig. 3</b>: Similarity of the decision surfaces between jet mass and five physicist engineered variables, as measured the discriminant ordering metric defined above. A zero indicates no similarity, a one indicates perfect similarity.
</div>

# Application: Reduce and orthogonalize
<div align="justify">
In the final paper, there are three different applications of the translation method described above to the problem of classifying collisions. The first approach attempts to explain the gap between neural nets trained on raw data and the performance of the high level engineered variables by searching for a new high level engineered variable that captures the information discrepancy. The second attempts to gain insight into the neural net strategy on the images by comparing the network output with the space of momentum fraction functions using the discriminant ordering. The third approach looks at the information content in the high level engineered variables by attempting to reduce and orthogonalized them into a minimal set. Since my work involved the last approach, this post will explore this strategy. (For more information on the first two approaches, see the paper.)

<br><br>

My work involved taking the set of high level engineered variables and reducing them to as small a set as possible such that they maintained the same classification power as the total set and were orthogonal with all the other high level variables. When we say that two variables are orthogonal, we mean that they don’t contain the same information about the example, specifically, that they have a small discriminant ordering between each other.

<br><br>

The procedure I used attempts to iteratively replace each of the high level engineered variables with a learned variable that captures its same classification power but minimizes the information overlap with the other variables (see figure 4). The steps for this process are:

<ol type="1">
  <li>Scan the list variables and select the one which has the largest negative impact on the classification power when removed from the set.</li>
	<br>
  <li>Replace the variable by a neural network (subnet) which takes as input the jet image and returns as output a single number, to be later identified as a potential new high level variable. The aggregate network now takes as input 5 high level variables and the output from the subnet (which takes as input the image of the collision) and tries classify the event.</li>
	<br>
  <li>Train this network to classify images. In parallel, train an adversarial network, which takes in the subnet output and attempts to learn the values of the 5 other high levels features used to train the first network. These two networks have adverse loss functions, which pushes the subnet to produce a value that both maximizes its classification strength and minimizes its similarity to the other high level variables.</li>
	<br>
  <li>Freeze the weights in the subnet from step 2. This subnet is will now be considered a high level variable for future iterations. However, it cannot be chosen for replacement in step 1.</li>
	<br>
  <li>Repeat this procedure by going back to step 1 until removing HL variables has no impact on the classification strength.</li>
</ol>  
</div>

<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig4.png"
       width="700">
</p>
<div align="justify">
  <b>Fig. 4</b>: Step 1 analyzes the output of a subnet with only jet image features into a single output and trained as part of a larger classification network which also has access to high level engineered features except for one feature which is being replaced; the subnet is also trained by an adversarial network, which penalizes it if the subnet output can be used to predict the other five high level features. The subnet output is also compared to the &tau; function space of possible strategies using the discriminant ordering metric. Step 2 analyzes the output of a second subnet, trained as shown to replace a second high level feature with an orthogonalized version.
</div>

<br>


<div align="justify">
I found that of the six high level variables,
</div>
<p><p> $$ M_{jet}, \ C_2^{\beta=2}, \ C_2^{\beta=1} $$ </p></p>
<div align="justify">
(in this order) were best able to be transformed so as to capture comparable classification information as the original six high level variables while being orthogonal to other variables. I used ROC curve AUCs as the metric for classification strength and discriminant ordering as the metric for orthogonality. The results are (note that hat indicates a learned, orthogonal variable):
</div>

* $$ M_{jet}, \ C_2^{\beta=1}, \ C_2^{\beta=2}, \ D_2^{\beta=1}, \ D_2^{\beta=2}, \ \tau_{21}^{\beta=1} $$ had an AUC of 0.946
<br>
* $$ \hat{M}_{jet}, \ \hat{C}_2^{\beta=1}, \ \hat{C}_2^{\beta=2}, \ D_2^{\beta=1}, \ D_2^{\beta=2}, \ \tau_{21}^{\beta=1} $$ had an AUC of 0.940
<br>
* $$ \hat{M}_{jet}, \ \hat{C}_2^{\beta=1}, \ \hat{C}_2^{\beta=2} $$ had an AUC of 0.906

<div align="justify">
A table listing the discriminant ordering between the three orthogonalized variables and the original high level variables is shown in figure 5.
</div>

<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/do_orthogonal.png"
       width="500">
</p>
<div align="justify">
  <b>Fig. 5</b>: Comparison of the similarity of classification by original high level variable and orthogonalized versions.
</div>

<br>

<div align="justify">
My worked showed some success in learning high level variables that can both classify and be orthogonal with the other variables. However, the results are not perfectly able to replicate the classifying power of the original high level variables and having zero similarity with the other variables.
	
<br>
</div>

After learning the 3 orthogonal, high level variables, I then compared them to $$\tau$$ functional space as previously defined. The comparison against the $$\tau_1$$ showed showed no telling results. Figure 6 shows the comparison between the orthogonal variables and $$\tau_2$$ at points in the parameter space.

<br>

<p align="center">
  <img src="//raw.githubusercontent.com/eweik/eweik.github.io/master/images/learning_physics_from_machine/fig6.png"
       width="1200">
</p>
<div align="justify">
  <b>Fig. 6</b>: Comparison of the three orthogonalized high level variables with points in the functional space defined by two parameters.
</div>

<br>

We see that $$ \hat{M}_{jet} $$ and $$ \hat{C}_2^{\beta=2} $$ have somewhat significant discriminant orderings with points in $$\tau_2$$ space.

<div align="justify">
We leave the interpretation of the above results for others. Note: we limit the search in parameter space as shown in the figures due to computational resources. Future work could include exploring larger parts of this space and spaces of higher dimensionality.
</div>

# Conclusion
<div align="justify">
Neural networks are strong. Often, better at tasks than expert humans. 
	
<br><br>

But, that doesn't mean that humans should just consider physics too hard and let these machines handle all the complicated things. We should be able to use these computers to help provide insight into these complex tasks in order to help augment our own understanding instead of surrendering modalities of thinking. 

In this post, I showed only a small method for how the output of neural networks can be interpreted for physicists to make sense of. But the research involved in neural network interpretation is wide ranging and only growing with time. Future work includes exploring these areas to understand what specifically the neural network used from the input to makes its decision and how each hidden layer effects that input to arrive at its decision. Much progress is still needed. Many questions remain. It's an exciting time in research.

<br>
</div>

The paper containing this work is currently being prepared for submission.
