% Imports and cleaning up
addpath 'util';
clear all; clc; close all;

% Folders that have to be set individually
data_root_folder = './';
scene_filter = 'F';

%---- IDEALLY THESE HYPERPARAMETERS CAN BE OPTIMIZED DURING TRAINING ------

% HYPERPARAMETERS OF THE MODEL
% Input size of our images - W x H x Color dimension
inputSize = [227 227 1];
% Number of hidden units in the RNN layers
numHiddenUnits = 2048;
% divide image into several grids of boxes and regress 
% confidene for each box in each grid 
gridSize = [6];
% widths and heights of anchor boxes
anchorBoxes = [gridSize
                gridSize]';
% Number of outputs of regression layer = number of grid boxes we have
numOutputs = 0;
for s = gridSize
   numOutputs = numOutputs + ceil(inputSize(1)/s)^2; 
end

% HYPERPARAMETERS FOR TRAINING
maxEpochs = 3;
miniBatchSize  = 32;
optimizer = 'adam';
initialLearnRate = 1e-4;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 6;
shuffleFrequency = 'every-epoch';
executionEnvironment = 'cpu';

% HYPERPARAMETERS FOR EVALUATION
% these hyperparameters you can play around with
% confidence threshold when a bounding box candidate is considered as an
% actual prediction
candidate_confidence = 0.025;
% IoU threshold for when a prediction counts as correct
iou_threshold = 0.01;

