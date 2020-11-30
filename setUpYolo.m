%%%% CONSTANTS and HYPERPARAMETERS %%%%%
% Input size of our images - W x H x Color dimension
inputSize = [454 454 1];
% number of anchor boxes
numAnchors = 6;

% anchor boxes
%rng(0)
%trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
%[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);
anchorBoxes = [41,34
   163,130
    98,93
   144,125
    33,24
    69,66];
area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
[~, idx] = sort(area, 'descend');
anchorBoxes = anchorBoxes(idx, :);
anchorBoxMasks = {[1,2,3]
    [4,5,6]
    };

%Feature extraction network
baseNetwork = squeezenet;
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

% specify detection output
networkOutputs = ["conv2Detection1"
    "conv2Detection2"
    ];

lgraph_yolo = lgraph;

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
    reluLayer('Name','relu1Detection2')
    convolution2dLayer(1,numFilters,'Padding','same','Name','conv2Detection2','WeightsInitializer','he')
    ];
lgraph = addLayers(lgraph,secondDetectionSubNetwork);
end
