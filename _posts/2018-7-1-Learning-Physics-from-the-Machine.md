---
layout: post
title: Learning Physics from the Machine
data: 2018-07-01 12:00:00 
category: 
tags:
---

An overview of an undergraduate research project.

<div align="justify">
In particle physics, people smash particles together at very high speeds to see what happens. The higher the speed, the more likely we can see something we’ve never seen before, a new and exciting discovery. For example, the Higgs Boson, which holds the secret to the nature of mass, was discovered in 2013 at the Large Hadron Collider (LHC). These types of experiments can help tell how us what are the most fundamental components of all matter.
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
However, new and exciting particles are rare. So we have to look at a lot of events (collisions) to possibly see something cool. This is why experiments like the LHC produce approximately 600 million events per second, which is then filtered to around 100-200 events/second for analysis. After some time, that’s a lot of data! 

<br><br>

To classify these events, physicists have come up with their own algorithms that take in the raw data of an event and produce what we will call “high-level engineered variables.” In theory, these high level variables should carry the same information as the raw, or “low level,” data. In this work, the high level variables we look at are:

</div>

<p><p> $$ M_{jet}, C_2^{\beta=1}, C_2^{\beta=2}, D_2^{\beta=1}, D_2^{\beta=2}, \tau_{21}^{\beta=1} $$ </p></p>

<div align="justify">

But, recent advances in machine learning has shown this to not necessarily be the case. In recent work by <a href="https://arxiv.org/abs/1603.09349">Baldi, et al. (2016)</a>, neural networks have actually outperformed high level engineered variables for tasks such as classification (figure 2). This suggests that the high level engineered variables do not capture the same information as the low level data and that the algorithms physicists use for these variables are not optimal. The question now becomes: what is the neural network doing differently?

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
This, however, is a tricky question to answer. Neural networks work well, but there are many open questions about them. Why do they work so well? What is the neural net learning from the data? Why is it learning that particular thing?

<br><br>

The objective of this work is to interpret what the neural network has learned in terms that physicists could understand. The motivation for this is to gain insight and possibly learn something new about the nature of the physics involved.
</div>

# Translating from machine to human
<div align="justify">
This is the tricky part. Taking the neural network output and translating it into something physicists can understand. 

<br><br>

For this particular work, we looked at the problem of classifying jets of particles using substructure information. Essentially, the neural network would take an image of the collision and classify it as either signal or background. The neural network makes these decisions using a set of numbers (weights) that it learns to transform the input into an output. Thus far, physicists can not make sense out of these weights and transformation; in fact, no one, not just physicists, can really understand these things with great insight. They just happen to work and that’s that.

<br><br>

So, we want to take the information from the neural net and translate it into something physicists can understand. Based on certain arguments and assumptions in quantum physics involving particle collisions which are beyond my scope, physicists can understand these jet images through functions involving the momentum fraction in each pixel in the jet. The first three terms in this family of functions is expressed as:

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
So we want to take the information from the neural net and compare it to these functions. If they are similar, this could suggest that the neural network is learning something related to the physics of these functions. Physicists could then explore these functions and possibly learn more about the nature of these collisions!

<br><br>

But how do we compare two functions? We need a metric that reflects the task of using the information from the low level data (image) for classification. However, it must also be invariant to non-linear 1-to-1 transformations of the two functions, because of the non-linear nature of neural networks. So, we consider the decision surfaces defined by the threshold on the function output. By decision surface, we mean the hyperplane which separates what the function believes to be signal and background. The space where the hyperplane is defined would depend on the function or neural net at hand; but the important part is the threshold for which is separates signal and background. Specifically, we consider two functions similar if, for any pair of points, they produce the same signal-to-background ordering. This is expressed through what we call the <i>discriminant ordering</i> between two functions, <i>f</i> and <i>g</i>, for a pair of points <i>x</i> and <i>x'</i>:

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
  <b>Fig. 3</b>: Similarity of the decision surfaces between jet mass and five physicist engineered variables, as measured the discriminant ordering metric defined above. A zero indicates no similarity, a one indicates perfect similarity.
</div>

# Application: Reduce and orthogonalize
<div align="justify">
Now there are a few different approaches our group took to apply the translation method described above to the problem of classifying collisions. The first approach attempts to explain the gap between neural networks trained on the images and the performance of the high level engineered variables by searching for a new high level engineered variable that captures the lost information. The second approach attempts to gain insight into the neural network strategy on the images by mapping it into our space of functions. The third approach looks at the information content in the high level engineered variables by attempting to reduce and orthogonalized them into a minimal set. Since my work involved the last approach, this post will explore this more deeply. (For more information on the first two approaches, see the paper.)

<br><br>

My approach was to take the set of high level engineered variables and reduce them to as small a set as necessary that both maintained the original classification power and was orthogonal within the other high level variables. By orthogonal, I mean that they don’t contain the same information about the event, that is, they have a small discriminant ordering between each other.

<br><br>

The procedure I used attempts to iteratively replace each of the high level engineered variables with a learned variable that captures its same classification power but minimizes the information overlap with the other variables (see figure 4). The steps for this process are:

<ol type="1">
  <li>Scan the list variables and select the one which has the largest negative impact on the classification power when it is removed from the set.</li>
	<br>
  <li>Replace the variable by a neural network (subnet) which takes as input the jet image and returns as output a single number, to be later identified as a potential new high level variable. This new network now contains a neural network which takes as input 5 high level variables and the output from the subnet (which takes as input the image of the collision) and tries classify the event.</li>
	<br>
  <li>Train this new network structure with an adversarial network, which uses the subnet output to attempt to recover the values of the other HL variables. These two networks have adverse loss functions, which pushes the subnet to produce a value that both maximizes its classification strength and its independence from the existing high level variables.</li>
	<br>
  <li>Freeze the weights in the subnet from step 2. This subnet is will now be considered a high level variable for future iterations. However, it cannot be chosen for replacement in step 1.</li>
	<br>
  <li>Repeat this procedure by going back to step 1 until removing HL variables has no impact.</li>
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
<p><p> $$ M_{jet}, C_2^{\beta=2}, C_2^{\beta=1} $$ </p></p>
<div align="justify">
(in this order) were best able to be transformed so as to capture the same classification information from the original six high level variables while being orthogonal to other variables. I used ROC curve AUC as the metric for the variables’ ability for classification ability and discriminant ordering as the metric for orthogonality. The results are (note that hat indicates a learned, orthogonal variable):
</div>

* $$ M_{jet}, C_2^{\beta=1}, C_2^{\beta=2}, D_2^{\beta=1}, D_2^{\beta=2}, \tau_{21}^{\beta=1} $$ had an AUC of 0.946
<br>
* $$ \hat{M}_{jet}, \hat{C}_2^{\beta=1}, \hat{C}_2^{\beta=2}, D_2^{\beta=1}, D_2^{\beta=2}, \tau_{21}^{\beta=1} $$ had an AUC of 0.940
<br>
* $$ \hat{M}_{jet}, \hat{C}_2^{\beta=1}, \hat{C}_2^{\beta=2} $$ had an AUC of 0.906

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
I show some success in learning high level variables that can both classify and be orthogonal with the other variables. However, the results are not perfect in the sense of perfectly replicating the classifying power of the original high level variables and having zero similarity with the other variables.
	
<br>
</div>

After learning our 3 orthogonal, high level variables, I then compared them to $$\tau$$ functional space as previously defined. The comparison against the $$\tau_1$$ showed showed no telling results. Figure 6 shows the comparison between the orthogonal variables and $$\tau_2$$ at points in the parameter space.

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
The interpretation of the above results is left for theorists. Note: we limit the search in parameter space as shown in the figures due to computational resources. Future work could include exploring larger parts of this space and spaces of higher dimensionality.
</div>

# Conclusion
<div align="justify">
Neural networks have powerful properties that often perform better at tasks that experts are not perfect at. This is even in the case of classifying collisions of particles at high speeds, which are particularly complex. 
	
<br><br>

I believe that neural networks can provide insight into these types of complex tasks. We showed how the output of neural networks can be interpreted for physicists to make sense of, for them to possibly learn something new about the physics involved in these collisions. But there is still work ahead. Many questions remain. Understanding the sensitivity to the input and understanding the roles of hidden nodes in the neural network are still ideas not settled. Future work includes exploring these areas to understand what specifically the neural network used from the input to makes its decision and how each hidden layer effects that input to arrive at its decision.

<br>
</div>

The paper containing this work is currently being prepared for submission.
