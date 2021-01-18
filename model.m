% The layers of our main model
layersPreCNN = [ ...
    %%%% INPUT PREP %%%%%
    % input layer - defines size of our pictures
    sequenceInputLayer(inputSize + [1 0 0],'Name','my_input_images','Normalization','zerocenter')
    % convert the sequences of images to an array of images
    sequenceFoldingLayer('Name','foldInput')
    ];
    
%%%% CNN %%%%%
% apply our CNN independently to each time step
% we define the CNN seperately
lgraph_cnn = setUpCNN();
    
%%%% RNN $$$$$
layersRNN = [...
    batchNormalizationLayer('Name','batchnormRNN')
    bilstmLayer(numHiddenUnits,'OutputMode','last','Name','bilstm')
    reluLayer('Name','reluRNN')
    dropoutLayer(0.5,'Name','dropoutRNN')
    %%% OUTPUT LAYERS %%%%%
    fullyConnectedLayer(numOutputs,'Name','fc')
    sigmoidLayer('Name','sigmoid')
    CELoss('ceLoss')];

% add pre CNN part and connect
net = addLayers(lgraph_cnn,layersPreCNN);

% add all additional layers
% layers for detection heads
unfoldLayer1 = sequenceUnfoldingLayer('Name','unfoldHead1');
flattenLayer1 = flattenLayer('Name','flattenHead1');
unfoldLayer2 = sequenceUnfoldingLayer('Name','unfoldHead2');
flattenLayer2 = flattenLayer('Name','flattenHead2');
net = addLayers(net,[unfoldLayer1
  flattenLayer1]);
net = addLayers(net,[unfoldLayer2
  flattenLayer2]);
% layers for poses
splitLayer = splitInputLayer('splitInput');
unfoldLayer3 = sequenceUnfoldingLayer('Name','unfoldPoses');
flattenLayer3 = flattenLayer('Name','flattenPoses');
net = addLayers(net,splitLayer);
net = addLayers(net,[unfoldLayer3
    flattenLayer3]);
% concatenation layer
concatLayer = concatenationLayer(1,3,'Name','concat');
net = addLayers(net,concatLayer);
% RNN
net = addLayers(net,layersRNN);

% connect the two output heads
net = connectLayers(net,'tanhDetection1','unfoldHead1/in');
net = connectLayers(net,'tanhDetection2','unfoldHead2/in');
net = connectLayers(net,'foldInput/miniBatchSize','unfoldHead1/miniBatchSize');
net = connectLayers(net,'foldInput/miniBatchSize','unfoldHead2/miniBatchSize');

% connect poses layers
net = connectLayers(net,'foldInput/out','splitInput');
net = connectLayers(net,'splitInput/out1','conv1');
net = connectLayers(net,'splitInput/out2', 'unfoldPoses/in');
net = connectLayers(net,'foldInput/miniBatchSize','unfoldPoses/miniBatchSize');

% connect concat layer
net = connectLayers(net,'flattenHead1','concat/in1');
net = connectLayers(net,'flattenHead2','concat/in2');
net = connectLayers(net, 'flattenPoses','concat/in3');

% connect RNN
net = connectLayers(net,'concat','batchnormRNN');

function lgraph = setUpCNN()
%Feature extraction network
baseNetwork = squeezenet();
lgraph = squeezenetFeatureExtractor(baseNetwork);

% Add detection heads to the feature extraction network
lgraph = addFirstDetectionHead(lgraph);
lgraph = addSecondDetectionHead(lgraph);
lgraph = connectLayers(lgraph, 'fire9-concat', 'batchnorm_head1');
lgraph = connectLayers(lgraph, 'relu1Detection1', 'batchnorm_head2');
lgraph = connectLayers(lgraph, 'fire5-concat', 'depthConcat1Detection2/in2');
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

function lgraph = addFirstDetectionHead(lgraph)
% The addFirstDetectionHead function adds the first detection head.

% Compute the number of filters for last convolution layer.
firstDetectionSubNetwork = [
    batchNormalizationLayer('Name','batchnorm_head1');
    convolution2dLayer(3,256,'Padding','same','Name','conv1Detection1','WeightsInitializer','he')
    reluLayer('Name','relu1Detection1')
    convolution2dLayer(1,4,'Padding','same','Name','conv2Detection1','WeightsInitializer','he')
    batchNormalizationLayer('Name','batchnormDetection1')
    fullyConnectedLayer(512,'Name','fcDetection1')
    tanhLayer('Name','tanhDetection1')
    ];
lgraph = addLayers(lgraph,firstDetectionSubNetwork);
end

function lgraph = addSecondDetectionHead(lgraph)
% The addSecondDetectionHead function adds the second detection head.

% Compute the number of filters for the last convolution layer.
    secondDetectionSubNetwork = [
    batchNormalizationLayer('Name','batchnorm_head2');
    upsampleLayer(2,'upsample1Detection2')
    depthConcatenationLayer(2, 'Name', 'depthConcat1Detection2');
    convolution2dLayer(3,128,'Padding','same','Name','conv1Detection2','WeightsInitializer','he')
    %averagePooling2dLayer(2,'Stride',2,'Name','avgPool')
    reluLayer('Name','relu1Detection2')
    convolution2dLayer(1,2,'Padding','same','Name','conv2Detection2','WeightsInitializer','he')
    batchNormalizationLayer('Name','batchnormDetection2')
    fullyConnectedLayer(1024,'Name','fcDetection2')
    tanhLayer('Name','tanhDetection2')
    ];
lgraph = addLayers(lgraph,secondDetectionSubNetwork);
end

