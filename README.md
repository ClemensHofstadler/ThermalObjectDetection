# Thermal Image Object Detection for Search and Rescue Operations

This is the Matlab Coded needed to reproduce the results described in [this blog post](https://clemenshofstadler.com/2021/02/21/thermal-image-object-detection-for-search-and-rescue-operations/). Note that this code requires the image processing, the computer vision as well as the deep learning toolbox. All code was designed for Matlab 2020b.

This work was a joint project done together with Jakob Geringer, Horst Gruber, and Shahed Masoudian.

## Setting up everything
To get everything ready, download the repository, unzip the <i>data</i> folder and put the <i>data</i> folder as well as all Matlab scripts into a directory where Matlab can find it.

## Training the model
If all Matlab scripts are in the same folder where also the <i>data</i> folder is, simply run

1. The <i>hyperparameters.m</i> script. This defines the basic hyperparameters which are needed for everything else (such as the number of hidden units in the network, the input and output size of the network or the gridsize to interpret the outputs).
2. The <i>model.m</i> script. This creates the model and saves it in a variable named <i>net</i>. You can analyze the network by calling <i>analyzeNetwork(net)</i>.
3. The <i>train.m</i> script. This loads and prepares the training data and specifies the hyperparameters for training. Note: loading and preparing the data takes a few minutes.
3. To then actually train the network, simply run the command <i>trained_network = trainNetwork(X,Y,net,options)</i>. This should open a second window, where the training process is visualized. After training, the trained network is saved in the variable <i>trained_network</i>.

## Evaluating the model
In order to evaluate the model on the test data, 
