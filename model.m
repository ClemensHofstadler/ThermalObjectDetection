%%%% HYPERPARAMETERS %%%%%
% Input size of our images - W x H x Color dimension
inputSize = [454 454 1];
% Dimension along which to concatenate the outputs
concateDim = 1;
% Number of hidden units in the RNN layers
numHiddenUnits1 = 100;
numHiddenUnits2 = 100;
% Number of outputs of regression layer 
numOutputs = 10;
% Number of features from the poses
numFeaturesPoses = 12;
% number of anchor boxes
numAnchors = 6;


% The layers of our main model
layersPreCNN = [ ...
    %%%% INPUT PREP %%%%%
    % input layer - defines size of our pictures
    sequenceInputLayer(inputSize,'Name','my_input_images')
    % convert the sequences of images to an array of images 
    sequenceFoldingLayer('Name','my_fold')];
    
    %%%% CNN %%%%%
    % apply our CNN independently to each time step
    % we define the CNN seperately
    
     %%%% RNN $$$$$
 layersRNN = [...
    bilstmLayer(numHiddenUnits1,'OutputMode','last','Name','my_bilstm1')
    batchNormalizationLayer('Name','my_batchNorm1')
    reluLayer('Name','my_relu1')
    bilstmLayer(numHiddenUnits2,'OutputMode','last','Name','my_bilstm2')
    batchNormalizationLayer('Name','my_batchNorm2')
    reluLayer('Name','my_relu2')
    
    %%% OUTPUT REGRESSION %%%%%
    fullyConnectedLayer(numOutputs,'Name','my_fc')
    regressionLayer('Name','my_regression')];


% add pre CNN part and connect
net = addLayers(lgraph_yolo,layersPreCNN);
net = connectLayers(net,'my_fold/out','conv1');

% connect the two output heads
unfoldLayer1 = sequenceUnfoldingLayer('Name','my_unfold_1');
flattenLayer1 = flattenLayer('Name','my_flatten_1');
unfoldLayer2 = sequenceUnfoldingLayer('Name','my_unfold_2');
flattenLayer2 = flattenLayer('Name','my_flatten_2');
net = addLayers(net,[unfoldLayer1
  flattenLayer1]);
net = connectLayers(net,'conv2Detection1','my_unfold_1/in');
net = addLayers(net,[unfoldLayer2
  flattenLayer2]);
net = connectLayers(net,'conv2Detection2','my_unfold_2/in');
net = connectLayers(net,'my_fold/miniBatchSize','my_unfold_1/miniBatchSize');
net = connectLayers(net,'my_fold/miniBatchSize','my_unfold_2/miniBatchSize');

% add concatenation layer
concatLayer = concatenationLayer(1,2,'Name','my_concat');
net = addLayers(net,concatLayer);
net = connectLayers(net,'my_flatten_1','my_concat/in1');
net = connectLayers(net,'my_flatten_2','my_concat/in2');

% add RNN part
net = addLayers(net,layersRNN);
net = connectLayers(net,'my_concat','my_bilstm1');

