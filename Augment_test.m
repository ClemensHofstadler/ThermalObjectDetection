addpath 'util';
clear all; clc; close all; %clean up!

%% SETUP
linenumber = 1;
site = 'F1';
datapath = fullfile( './data/', site );
thermalParams = load( './data/camParams_thermal.mat' ); %load intrinsics
thermalpath = fullfile( datapath, 'Images', num2str(linenumber) ); % path to thermal images
thermalds = datastore( thermalpath );
%% scaling Test

thermal = readimage( thermalds, 16 );  
m = size(thermal);
figure( 'Color', 'white' ); clf;
subplot(2,1,1); imshow( thermal, [] );
title( 'original' );
thermal = undistortImage( thermal, thermalParams.cameraParams );
thermal=double(thermal);
thermal = thermal./max(max(thermal));
thermal = imresize(thermal,[454 454]); 
ax2 = subplot(2,1,2); imshow( thermal, [] );
colormap( ax2, 'parula' );
title( 'undistorted + colormap' );

%% display multiple images of a line
imgIds = [14:18];
figure( 'Color', 'white' ); clf;
for i = 1:length(imgIds)
    thermal = undistortImage( readimage( thermalds, imgIds(i) ), thermalParams.cameraParams );
    subplot(1,length(imgIds),i); imshow( thermal, [] );
end
colormap( 'parula' );
%% load poses Flipped and scaled Camera Parameter Test
json = readJSON( fullfile( datapath, '/Poses/', [num2str(linenumber) '.json'] ) );
images = json.images; clear json;
K = thermalParams.cameraParams.IntrinsicMatrix; % intrinsic matrix, is the same for all images
Ms = {};
MS = {};
figure('Color','white','Name','Poses'); hold on;
Rm = [-1 0 0;...
      0 1 0;...
      0 0 1];
Rs = [0.88671875 0 0;...
      0 0.709375 0;...
      0 0 1];
k = K;
for i_label = 1:length(imgIds)
   M = images(imgIds(i_label)).M3x4;% read the pose matrix
   M(4,:) = [0,0,0,1];
   Ms{i_label} = M;
   invM = inv(M);
   pos = M(:,4);
   cam = plotCamera( 'Location', pos(1:3), 'Size', .2 ); hold on;
end
axis equal
axis off
 
figure('Color','white','Name','Poses'); hold on;
for i_label = 1:length(imgIds)
   iM = images(imgIds(i_label)).M3x4;% read the pose matrix
   MM = Rs*Rm*iM; %transform with matrix rotation.
   MM(4,:) = [0,0,0,1];
   MS{i_label} = MM;
   invMM = inv(MM);
   POS = MM(:,4);
   CAM = plotCamera( 'Location', POS(1:3), 'Size', .2 ); hold on;

end
axis equal
axis off
%% Testing fliped image Matrix Validity Via Integral
integral = zeros(size(thermal),'double');
count = zeros(size(integral),'double');

% warp to a reference image (center view)
M1 = Ms{3};
R1 = M1(1:3,1:3)';
t1 = M1(1:3,4)';

for i = 1:length(imgIds)
    img1 = undistortImage( imread(fullfile(thermalpath,images(imgIds(i)).imagefile)), ...
           thermalParams.cameraParams );      
    M2 = Ms{i};
    R2 = M2(1:3,1:3)';
    t2 = M2(1:3,4)';

    % relative 
    R = R1' * R2;
    t = t2 - t1 * R;

    z = 30; %getAGL(site); % meter
    P = (inv(K) * R * K );
    P_transl =  (t * K);
    P(3,:) = P(3,:) + P_transl./z; % add translation
    tform = projective2d( P );

    % --- warp images ---
    warped2 = double(imwarp(img1,tform.invert(), 'OutputView',imref2d(size(integral))));
    warped2(warped2==0) = NaN; % border introduced by imwarp are replaced by nan
    
    figure(8);
    subplot(1,length(imgIds),i); imshow( warped2, [] );

    count(~isnan(warped2)) = count(~isnan(warped2)) + 1;
    integral(~isnan(warped2)) = integral(~isnan(warped2)) + warped2(~isnan(warped2));
end

colormap( 'parula' );
% normalize
integral = double(integral) ./ count;

h_fig = figure(9);
set( h_fig, 'Color', 'white' ); clf;
imshow( integral, [] ); title( 'Warped integral' );
Integral = zeros([454 454],'double');
Count = zeros(size(Integral),'double');
M11 = MS{3};
R11 = M11(1:3,1:3)';
t11 = M11(1:3,4)';

for i = 1:length(imgIds)
    Thermal = undistortImage( imread(fullfile(thermalpath,images(imgIds(i)).imagefile)), ...
           thermalParams.cameraParams );
    Thermal=double(Thermal);
    Thermal = Thermal./max(max(Thermal));
    img1 = flipdim(Thermal ,2);
    img2 = imresize(img1,[454 454]);
    %img1 = flipdim(Thermal ,1);  
    %img2 = flipdim(img1 ,2);
    M2 = MS{i};
    R2 = M2(1:3,1:3)';
    t2 = M2(1:3,4)';

    % relative 
    R = R11' * R2;
    t = t2 - t11 * R;
   
    z = Rs(1,1)*30; %getAGL(site); % meter
    P = (inv(k) * R * k ); 
    P_transl =  (t * k);
    P(3,:) = P(3,:) + P_transl./z; % add translation
    tform = projective2d( P );

    % --- warp images ---
    warped1 = double(imwarp(img2,tform.invert(), 'OutputView',imref2d(size(Integral))));
    warped1(warped1==0) = NaN; % border introduced by imwarp are replaced by nan
    
    figure(10);
    subplot(1,length(imgIds),i); imshow( warped1, [] );

    Count(~isnan(warped1)) = Count(~isnan(warped1)) + 1;
    Integral(~isnan(warped1)) = Integral(~isnan(warped1)) + warped1(~isnan(warped1));
end
colormap( 'parula' );

% normalize
Integral = Integral ./ Count;
h_fig = figure(11);
set( h_fig, 'Color', 'white' ); clf;
subplot(1,2,1)
imshow( Integral, [] ); title( 'Scaled Warped Integral' );
subplot(1,2,2)
imshow( integral, [] ); title( 'Normal Image Warped Integral' );
%%
figure( 'Color', 'white' ); clf;
for i = 1:length(imgIds)
    thermal = undistortImage( readimage( thermalds, imgIds(i) ), thermalParams.cameraParams );
    I2 = flipdim(thermal ,2);
    subplot(1,length(imgIds),i); imshow( I2, [] );
    %subplot(1,length(imgIds),i); imshow( thermal, [] );
end
%colormap( 'parula' );
%%

%% how to flip image
I2 = flipdim(thermal ,2);           %# horizontal flip
I3 = flipdim(thermal ,1);           %# vertical flip
I4 = flipdim(I3, 2);    %# horizontal+vertical flip
subplot(2,2,1), imshow(thermal, []); title( 'Original' );
subplot(2,2,2), imshow(I2, []); title( 'flipped Horizontal' )
subplot(2,2,3), imshow(I3, []); title( 'flipped Vertical' )
subplot(2,2,4), imshow(I4, []); title( 'flipped Ver-Hor' )