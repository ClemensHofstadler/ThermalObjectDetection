# Thermal Image Object Detection for Search and Rescue Operations

This is the Matlab Coded needed to reproduce the results described in this [blog post](https://clemenshofstadler.com/2021/02/21/thermal-image-object-detection-for-search-and-rescue-operations/). Note that this code requires the image processing, the computer vision as well as the deep learning toolbox. All code was designed for Matlab 2020b.

This work was a joint project done together with Jakob Geringer, Horst Gruber, and Shahed Masoudian.

## Setting up everything
First, the data has to be downloaded from this [link](https://drive.google.com/file/d/1VJQQWp_0RUjmA5-fjix6rP79KWkP4NSS/view?usp=sharing). Unzip the downloaded folder and place the <i>data</i> folder in a directory where Matlab can find it. We also added our model to this download if someone is interested in reproducing our results. After downloading the data, download this repository and put all Matlab scripts into the same directory as the <i>data</i> folder.

## Training the model
If all Matlab scripts are in the same folder where also the <i>data</i> folder is, simply run

1. The <i>hyperparameters.m</i> script. This defines all hyperparameters which are needed for everything else (such as the hyperparameters of the network, the hyperparameters for training, and the hyperparameters for evaluation).
2. The <i>model.m</i> script. This creates the model and saves it in a variable named <i>net</i>. You can analyze the network by calling <i>analyzeNetwork(net)</i>.
3. The <i>train.m</i> script. This loads and prepares the training data. Note: loading and preparing the data takes a few minutes.
3. To then actually train the network, simply run the command <i>trained_network = trainNetwork(X,Y,net,options)</i>. This should open a second window, where the training process is visualized. After training, the trained network is saved in the variable <i>trained_network</i>.

## Evaluating the model
In order to evaluate the model on the test data, simply safe the trained network in a variable named <i>trained_net</i> and run the <i>evaluationScript.m</i> script. This then evaluates the model on the test data, plots a precison-recall curve, and prints some basic stats such as average precision, the number of true positives and the number of false positives. Note: the evaluation might take some time.
