% --------------- Example How to Load and Work with our thermal data ------
% this script computes integrals for every line and writes it into results.
% Additionally labels as AABBs are stored in text files. 
addpath 'util'
clear all; clc; close all; % clean up!

%%
%trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11' }; % Note, we use the same IDs as in the Nature Machine Intelligence Paper.
%testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};
trainingsites = { 'F0'};%, 'F1', 'F2', 'F3','F5'}; % Note, we use the same IDs as in the Nature Machine Intelligence Paper.
testsites = { 'T1'};%, 'T2', 'T3', 'T4',};
%allsites = cat(2, trainingsites, testsites );

%%Training 
[training_data] = dta_loader(trainingsites);
%%test data 
%[test_data] = dta_loader(testsites);

function [data] = dta_loader(g_site)

for i_site = 1:length(g_site)
site = g_site{i_site};
datapath = fullfile( './data/', site ); 
if ~isfolder(fullfile( datapath ))
   error( 'folder %s does not exist. Did you download additional data?', datapath );
end

resultsfolder = fullfile( './results/', site );
mkdir(resultsfolder);

thermalParams = load( './data/camParams_thermal.mat' );

%%
R2 =[-1 0 0;0 1 0;0 0 1];
R3 =[1 0 0;0 -1 0;0 0 1];
R4 =[-1 0 0;0 -1 0;0 0 1];
Rs = [0.88671875 0 0;0 0.709375 0;0 0 1]; 
% Note: line numbers might not be consecutive and they don't start at index
% 1. So we loop over the posibilities:
F_count = 0;
for linenumber = 1:99
    
    if ~isfile(fullfile( datapath, '/Poses/', [num2str(linenumber) '.json'] ))
        continue % SKIP!
    end
    
    json = readJSON( fullfile( datapath, '/Poses/', [num2str(linenumber) '.json'] ) );
    images = json.images; clear json;
    
    try
        json = readJSON( fullfile( datapath, '/Labels/', ['Label' num2str(linenumber) '.json'] ) );
        labels = json.Labels; clear json;
    catch
       warning( 'no Labels defined!!!' ); 
       labels = []; % empty
    end
    
    K = thermalParams.cameraParams.IntrinsicMatrix;
    K=Rs*K;% intrinsic matrix, is the same for all images
    Ms = {};

    thermalpath = fullfile( datapath, 'Images', num2str(linenumber) );
    
    data_count = 4*(F_count)+1;
    for i_label = 1:length(images)
       thermal = undistortImage( imread(fullfile(thermalpath,images(i_label).imagefile)), ...
           thermalParams.cameraParams );
       
       thermal=double(thermal);
       thermal = thermal./max(max(thermal));
       
       %Normalize Pics
       I1 = imresize(thermal,[454 454]);     %resize pics
       figure(1)
       img = imshow( I1, [] );
       %set(gcf,'position',[0,10,350,350])
       
       I2 = flipdim(I1 ,2);  %# horizontal flip
       %figure(2)
       %img = imshow( I2, [] );
       %set(gcf,'position',[0,10,350,350])
       
       I3 = flipdim(I1 ,1);
       %figure(3)
       %img = imshow( I3, [] );%# vertical flip
       %set(gcf,'position',[0,10,350,350])
       
       I4 = flipdim(I3, 2);
       %figure(4)
       %img = imshow( I4, [] );%# horizontal+vertical flip
       %set(gcf,'position',[0,10,350,350])
       
       M1 = Rs*images(i_label).M3x4;
       M2 = Rs*R2*M1; M3 = Rs*R3*M1;M4 = Rs*R4*M1;
       %Rs is scaling factor for resize and R is rotation matrix for flipping
       img_data(:,:,data_count,i_label) = I1;
       cam_param(:,:,data_count,i_label) = M1;
       img_data(:,:,data_count+1,i_label) = I2;
       cam_param(:,:,data_count+1,i_label) = M2;
       img_data(:,:,data_count+2,i_label) = I3;
       cam_param(:,:,data_count+2,i_label) = M3;
       img_data(:,:,data_count+2,i_label) = I4;
       cam_param(:,:,data_count+3,i_label) = M3;
       M(4,:) = [0,0,0,1];
       Ms{i_label} = M;
    end
    F_count=F_count+1;

end
data(i_site).image = img_data;
data(i_site).cam_param = img_data;
end
end


