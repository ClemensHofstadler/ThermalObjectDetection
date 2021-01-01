% The layers of our main model
layersPreCNN = [ ...
    %%%% INPUT PREP %%%%%
    % input layer - defines size of our pictures
    sequenceInputLayer(inputSize,'Name','my_input_images')
    % convert the sequences of images to an array of images
    sequenceFoldingLayer('Name','my_fold')
    ];
    
%%%% CNN %%%%%
% apply our CNN independently to each time step
% we define the CNN seperately
lgraph_yolo = setUpYolo(anchorBoxes);
    
%%%% RNN $$$$$
layersRNN = [...
    bilstmLayer(numHiddenUnits1,'Name','my_bilstm1')
    reluLayer('Name','my_relu1')
    dropoutLayer(0.2,'Name','my_drop1')
    bilstmLayer(numHiddenUnits2,'OutputMode','sequence','Name','my_bilstm2')
    reluLayer('Name','my_relu2')
    dropoutLayer(0.2,'Name','my_drop2')
    
    %%% OUTPUT REGRESSION %%%%%
    fullyConnectedLayer(numOutputs,'Name','my_fc')
    sigmoidLayer('Name','my_sigmoid')
    regressionLayer('Name','my_regression')];

% add pre CNN part and connect
net = addLayers(lgraph_yolo,layersPreCNN);

% add all additional layers
% layers for poses
splitLayer = splitInputLayer('splitInput');
unfoldLayer3 = sequenceUnfoldingLayer('Name','my_unfold_3');
flattenLayer3 = flattenLayer('Name','my_flatten_3');
net = addLayers(net,splitLayer);
net = addLayers(net,[unfoldLayer3
    flattenLayer3]);
% concatenation layer
concatLayer = concatenationLayer(1,3,'Name','my_concat');
net = addLayers(net,concatLayer);
% RNN
net = addLayers(net,layersRNN);

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

% connect poses layers
net = connectLayers(net,'my_fold/out','splitInput');
net = connectLayers(net,'splitInput/out1','conv1');
net = connectLayers(net,'splitInput/out2', 'my_unfold_3/in');
net = connectLayers(net,'my_fold/miniBatchSize','my_unfold_3/miniBatchSize');

% connect concat layer
net = connectLayers(net,'my_flatten_1','my_concat/in1');
net = connectLayers(net,'my_flatten_2','my_concat/in2');
net = connectLayers(net, 'my_flatten_3','my_concat/in3');

% connect RNN
net = connectLayers(net,'my_concat','my_bilstm1');


function yolo = setUpYolo(anchorBoxes)
% anchor boxes
%rng(0)
%trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
%[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);
area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
[~, idx] = sort(area, 'descend');
anchorBoxes = anchorBoxes(idx, :);
anchorBoxMasks = {[1]
    [2]};

%Feature extraction network
baseNetwork = squeezenet();
lgraph = squeezenetFeatureExtractor(baseNetwork);

classNames = {'person'};
numClasses = size(classNames, 2);
numPredictorsPerAnchor = 5 + numClasses;

% Add detection heads to the feature extraction network
lgraph = addFirstDetectionHead(lgraph, anchorBoxMasks{1}, numPredictorsPerAnchor);
lgraph = addSecondDetectionHead(lgraph, anchorBoxMasks{2}, numPredictorsPerAnchor);
lgraph = connectLayers(lgraph, 'fire9-concat', 'conv1Detection1');
lgraph = connectLayers(lgraph, 'relu1Detection1', 'upsample1Detection2');
lgraph = connectLayers(lgraph, 'fire5-concat', 'depthConcat1Detection2/in2');


yolo = lgraph;
end

function lgraph = squeezenetFeatureExtractor(net)
% The squeezenetFeatureExtractor function removes the layers after 'fire9-concat'
% in SqueezeNet and also removes any data normalization used by the image input layer.

% Convert to layerGraph.
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'data','drop9' 'conv10' 'relu_conv10' 'pool10' 'prob' 'ClassificationLayer_predictions'});
conv1Layer = convolution2dLayer(3,64,'Stride',2,'Padding','same','Name','conv1');
lgraph = replaceLayer(lgraph,'conv1',conv1Layer);
end

function lgraph = addFirstDetectionHead(lgraph,anchorBoxMasks,numPredictorsPerAnchor)
% The addFirstDetectionHead function adds the first detection head.

numAnchorsScale1 = size(anchorBoxMasks, 2);
% Compute the number of filters for last convolution layer.
numFilters = numAnchorsScale1*numPredictorsPerAnchor;
firstDetectionSubNetwork = [
    convolution2dLayer(3,256,'Padding','same','Name','conv1Detection1','WeightsInitializer','he')
    reluLayer('Name','relu1Detection1')
    convolution2dLayer(1,numFilters,'Padding','same','Name','conv2Detection1','WeightsInitializer','he')
    ];
lgraph = addLayers(lgraph,firstDetectionSubNetwork);
end

function lgraph = addSecondDetectionHead(lgraph,anchorBoxMasks,numPredictorsPerAnchor)
% The addSecondDetectionHead function adds the second detection head.

numAnchorsScale2 = size(anchorBoxMasks, 2);
% Compute the number of filters for the last convolution layer.
numFilters = numAnchorsScale2*numPredictorsPerAnchor;
    
secondDetectionSubNetwork = [
    upsampleLayer(2,'upsample1Detection2')
    depthConcatenationLayer(2, 'Name', 'depthConcat1Detection2');
    convolution2dLayer(3,128,'Padding','same','Name','conv1Detection2','WeightsInitializer','he')
    averagePooling2dLayer(2,'Stride',2,'Name','avgPool')
    reluLayer('Name','relu1Detection2')
    convolution2dLayer(1,numFilters,'Padding','same','Name','conv2Detection2','WeightsInitializer','he')
    ];
lgraph = addLayers(lgraph,secondDetectionSubNetwork);
end

