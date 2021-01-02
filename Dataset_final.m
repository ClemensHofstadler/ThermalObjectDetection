%% First run load_all_scenes.m file to get scenestructure
% save this file as .mat and put it in data directory along side
% cameraParams
%
% this code distord images, scale them , Augment them and save concatenate
% them 

%DATA(num of folder).seq.(num of data).(image,cam_param,poly)
%image gives data of the image 227*227
%cam_param gives relative poses
%poly gives label location (4*2*num of poly)

% depending on image number of Poly may be different. 

%addpath 'util'
clear all; clc; close all; % clean up!

%%
%load scene_struct.mat from data directory
Folder = load( './data/scene_struct.mat' ).scene_struct;
%% Testing on 1 data
clc
clear folder
for i =1:2
    folder(i) = Folder(i);
end
X = folder(2).seq(2).labels_rect(1).labels(7).poly;

if X(1,1)== X(2,1) || X(1,2)==X(3,2)
           X(1,1)=X(1,1)+1;X(4,1)=X(4,1)+1;
           X(2,1)=X(2,1)-1;X(3,1)=X(3,1)-1;
           X(1,2)=X(1,2)+1;X(2,2)=X(2,2)+1;
           X(3,2)=X(3,2)-1;X(4,2)=X(4,2)-1;
end 
if any(X(:,1))==0 
           X(1,1)=X(1,1)+0.02;X(4,1)=X(4,1)+0.02;
           X(2,1)=X(2,1)+0.01;X(3,1)=X(3,1)+0.01;
end 
if any(X(:,2))==0 
           X(1,2)=X(1,2)+0.02;X(2,2)=X(2,2)+0.02;
           X(3,2)=X(3,2)+0.01;X(4,2)=X(4,2)+0.01;
end 
X
polyin = polyshape({X(:,1)},{X(:,2)});
[pol_x,pol_y] = boundingbox(polyin,1);
thermalParams = load( './data/camParams_thermal.mat' );
cam_param = thermalParams.cameraParams;
imshow(undistortImage(folder(2).seq(2).images(1).data,thermalParams.cameraParams),[])
drawpolygon( 'Position', X );

%%
clc
[DATA] = dta_loader(folder);

function [data] = dta_loader(Folder)
thermalParams = load( './data/camParams_thermal.mat' );
K = thermalParams.cameraParams.IntrinsicMatrix;
cam_param = thermalParams.cameraParams;
R1 =[1 0 0;0 1 0;0 0 1];
R2 =[-1 0 0;0 1 0;0 0 1];
R3 =[1 0 0;0 -1 0;0 0 1];
R4 =[-1 0 0;0 -1 0;0 0 1];
Rs = [227/640 0 0;0 227/512 0;0 0 1]; %scaling images causes Intrinsic changes

for i_site = 1:length(Folder)


 
% Note: line numbers might not be consecutive and they don't start at index
% 1. So we loop over the posibilities:
F_count = 0;
for linenumber = 1:length(Folder(i_site).seq)
    
    images = Folder(i_site).seq(linenumber).images;
    poses = Folder(i_site).seq(linenumber).relative_poses; %poses are inside .M3x4
    labels = Folder(i_site).seq(linenumber).labels_rect; % poly is inside .labels, .poly 
    K=Rs*K;% intrinsic matrix, is the same for all images

    data_count = 4*(F_count)+1;
    for i_label = 1:length(images)
       thermal = undistortImage( images(i_label).data,cam_param);
       thermal = double(thermal);
       thermal = thermal./max(max(thermal));
       lbl = labels(i_label).labels;
       
       %Normalize Pics
       I1 = imresize(thermal,[227 227]);     %resize pics
       %figure(1)
       %img = imshow( I1, [] );
       I2 = flip(I1 ,2);  %# horizontal flip
       I3 = flip(I1 ,1);
       I4 = flip(I3, 2);
       %figure(4)
       %img = imshow( I4, [] );%# horizontal+vertical flip
       %set(gcf,'position',[0,10,350,350])
       M1 = R1*poses(i_label).M3x4;
       M2 = R2*M1; M3 = R3*M1;M4 = R4*M1;
       %R is rotation matrix for flipping
       i_site
       data_count
       i_label
       data(i_site).seq(data_count).image(i_label).dat = I1;
       data(i_site).seq(data_count).cam_param(i_label).dat = M1;
       data(i_site).seq(data_count).poly(i_label).dat = Labels(I1,lbl,1,K);
       data(i_site).seq(data_count+1).image(i_label).dat = I2;
       data(i_site).seq(data_count+1).cam_param(i_label).dat = M2;
       data(i_site).seq(data_count+1).poly(i_label).dat = Labels(I2,lbl,2,K);
       data(i_site).seq(data_count+2).image(i_label).dat = I3;
       data(i_site).seq(data_count+2).cam_param(i_label).dat = M3;
       data(i_site).seq(data_count+2).poly(i_label).dat = Labels(I3,lbl,3,K);
       data(i_site).seq(data_count+3).image(i_label).dat = I4;
       data(i_site).seq(data_count+3).cam_param(i_label).dat = M4;
       data(i_site).seq(data_count+3).poly(i_label).dat = Labels(I4,lbl,4,K);

    end
    
    F_count=F_count+1;

   % Labels(site,thermalpath,thermalParams,images,Ms1,1,K);
end
end
clear Folder;
end

function [POLY] = Labels(image,label,flip_direction,K)
    
    K_ = K; K_(4,4) = 1.0; % make sure intrinsic is 4x4

    % draw polygon
    POLY = zeros(4,2,length(label));
    figure(1)
    imshow(image,[])
    for i_label = 1:length(label)
       
       if isempty(label) || isempty(label(i_label).poly)
           continue;
       end
       
       x = label(i_label).poly(:,1);
       y = label(i_label).poly(:,2);
       
       if all(x)==0 && all(y)==0
           continue
       end
       if x(1)==x(2) || y(1)==y(3)
           x(1)=x(1)+1;x(4)=x(4)+1;
           x(2,1)=x(2)-1;x(3,1)=x(3)-1;
           y(1)=y(1)+1;y(2)=y(2)+1;
           y(3)=y(3)-1;y(4)=y(4)-1;
       end
       if any(x)==0 
           x(1)=x(1)+0.02;x(4)=x(4)+0.02;
           x(2,1)=x(2)+0.01;x(3,1)=x(3)+0.01;
       end
       if any(y)==0 
           y(1)=y(1)+0.02;y(2)=y(2)+0.02;
           y(3)=y(3)+0.01;y(4)=y(4)+0.01;
       end
        
       
       polyin = polyshape({x},{y});
       [pol_x,pol_y] = boundingbox(polyin,1);
       poly_x = [pol_x(1);pol_x(1);pol_x(2);pol_x(2)];
       poly_y = [pol_y(1);pol_y(2);pol_y(2);pol_y(1)];
       
       
       %change the scale of label position
       moved_POS = [(227/640).*poly_x,... 
                    (227/512).*poly_y];
       if flip_direction == 1
           poly = moved_POS;
       end
       if flip_direction == 2
           new_pos = (227/2)*[1;1;1;1 ]-moved_POS(:,1);
           poly = [(227/2)*[1;1;1;1]+new_pos,moved_POS(:,2)];
       end
       if flip_direction == 3
           new_pos = (227/2)*[1;1;1;1 ]-moved_POS(:,2);
           poly = [moved_POS(:,1),(227/2)*[1;1;1;1]+new_pos];
       end
       if flip_direction == 4
           new_pos = (227/2)*[1 1;1 1;1 1;1 1]-moved_POS;
           poly = (227/2)*[1 1;1 1;1 1;1 1]+new_pos;
       end
       %lbl(data_count).poly = poly;    
       %poly = label(i_label).poly;
       POLY(:,:,i_label)= poly;
       drawpolygon( 'Position', poly );
       title(num2str(i_label));
       hold on
    end
    hold off
    
    
end
    

