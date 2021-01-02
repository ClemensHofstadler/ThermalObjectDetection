# Computer-Vision

## Scaling 
all picture was scaled to 454*454 in order to be fed to the system and K which is matrix intrinsic was also scaled by the same amount , other than that Z which was used to calculate the transfer of the data was also scaled by the same amount. 
##  Augmentation
looking at the data it was found the sequence of data is changing with each subfolder exisiting in each folder therefor Augmentation was only done with Flipping image namely, Horizontal, Vertial and Verti-horizontal Flips. 
in order to compennsate for the effect of flipping new Cam parameters considering Rotation was introduced to the system . 

## Training our first model
In order to train our first model, you need the files
- hyperparameters.m
- model.m
- upsampleLayer.m
- splitInputLayer.m
- train.m

Just put all of these files into the same folder where also the "data" folder is, then run
1. <b> The "hyperparameters.m" script</b>. This defines some basic hyperparameters which are needed for everything else.
2. <b>The "model.m" script</b>. This creates our network and saves it in a variable named <i>net</i>. You can analyze the network by calling <i>analyzeNetwork(net)</i>.
3. <b> The "train.m" script</b>: This loads and prepares the data and specifies the hyperparameters for training. Note: loading and preparing the data takes around 3 mins.
3. To actually train the network, you then have to run <b>trained_network = trainNetwork(X,Y,net,options)</b>.

<b>I am pretty sure that you will need the Deep Learning toolbox</b> (https://www.mathworks.com/products/deep-learning.html) in order to run some of this code. 
By setting the <i>executionEnvironment</i> variable in the trainingOptions, you can also specify whether to train on CPU or on GPU. Here is further information about this: https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html.

When you run the <b>trained_network = trainNetwork(X,Y,net,options)</b> command, a second window should open where you can see the training progress. 

Just as a reference, it takes about 3min and 14.7GB RAM to do one forward and backward pass with a minibatch of size 8 on my laptop (on a single CPU).
