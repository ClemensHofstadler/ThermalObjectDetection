% Imports and cleaning up
addpath 'util';
%clear all; clc; close all;

% Folders that have to be set individually
data_root_folder = './';
scene_filter = 'F';


%---- IDEALLY THESE HYPERPARAMETERS CAN BE OPTIMIZED DURING TRAINING ------
% Input size of our images - W x H x Color dimension (2nd color dimension
% to 'hide' the relative poses)
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
