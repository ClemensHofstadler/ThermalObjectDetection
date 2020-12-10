% FIRST: load/setup the net by running the model.m script
% THEN: rund this script to train the net

% these parameters are fixed - changing them requires more work
gridSize = 20;
inputSize = [454 454 2];

% these parameters you can play around with
%trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11' };
trainingsites = {'F0','F1'};
maxEpochs = 1;
miniBatchSize  = 8;
optimizer = 'adam';
initialLearnRate = 1e-3;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 20;
shuffleFrequency = 'every-epoch';
executionEnvironment = 'auto';

[X,Y] = loadData(trainingsites,inputSize,gridSize);

options = trainingOptions(optimizer, ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',initialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',learnRateDropFactor, ...
    'LearnRateDropPeriod',learnRateDropPeriod, ...
    'Shuffle',shuffleFrequency, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment',executionEnvironment,...
    'Verbose',false);

% run this command to train the network
% trained_net = trainNetwork(X,Y,net,options);

function labelGrid = generateLabelGrid(label,inputSize,gridSize)
    % set up the label grid
    boxes_per_line = floor(inputSize(1) / gridSize);
    labelGrid = zeros(boxes_per_line);
    for i_label = 1:size(label,3)
        x_min = min(label(:,1,i_label));
        x_max = max(label(:,1,i_label));
        y_min = min(label(:,2,i_label));
        y_max = max(label(:,1,i_label));
        for i = 1:boxes_per_line
            for j = 1:boxes_per_line
                left_right_inside_box = floor(x_min/gridSize) <= i-1 && i-1 < ceil(x_max/gridSize);
                up_down_inside_box = floor(y_min/gridSize) <= j-1 && j-1 < ceil(y_max/gridSize);  
                if left_right_inside_box && up_down_inside_box
                    labelGrid(i,j) = 1;
                end
            end
        end
    end
    labelGrid = labelGrid(:);
end

function [X,Y] = loadData(sites,inputSize,gridSize)
    training_data = dta_loader(sites);
    X = {};
    Y = [];
    for i = 1:length(training_data)
        folder_images = training_data(i).image;
        folder_poses = training_data(i).cam_param;
        seqs = cell(size(folder_images,3),1);
        labels = zeros(size(folder_images,3),floor(inputSize(1) / gridSize)^2);
        for j = 1:size(folder_images,3)
            length_seq = dir(['./data/', sites{i}, '/Images/', num2str(ceil(j/4)), '/*.tiff']);
            length_seq = numel(length_seq);
            current_seq = zeros([inputSize,[length_seq]]);
            current_seq(:,:,1,:) = folder_images(:,:,j,1:length_seq);
            current_seq(1:3,1:4,2,:) = folder_poses(:,:,j,1:length_seq);
            seqs{j} = current_seq;
            labels(j,:) = generateLabelGrid(training_data(i).lblpos(j).pos,inputSize,gridSize);
        end
        X = [X
            seqs];
        Y = [Y
            labels];
    end
end

function [data] = dta_loader(g_site)

for i_site = 1:length(g_site)
site = g_site{i_site};
datapath = fullfile( './data/', site ); 
if ~isfolder(fullfile( datapath ))
   error( 'folder %s does not exist. Did you download additional data?', datapath );
end

%resultsfolder = fullfile( './results/', site );
%mkdir(resultsfolder);

thermalParams = load( './data/camParams_thermal.mat' );

%%
R1 =[1 0 0;0 1 0;0 0 1];
R2 =[-1 0 0;0 1 0;0 0 1];
R3 =[1 0 0;0 -1 0;0 0 1];
R4 =[-1 0 0;0 -1 0;0 0 1];
Rs = [454/640 0 0;0 454/512 0;0 0 1]; 
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
    Ms1 = {};Ms2 = {};Ms3 = {};Ms4 = {};

    thermalpath = fullfile( datapath, 'Images', num2str(linenumber) );
    
    data_count = 4*(F_count)+1;
    for i_label = 1:length(images)
       thermal = undistortImage( imread(fullfile(thermalpath,images(i_label).imagefile)), ...
           thermalParams.cameraParams );
       
       thermal=double(thermal);
       thermal = thermal./max(max(thermal));
       
       %Normalize Pics
       I1 = imresize(thermal,[454 454]);     %resize pics
       %figure(1)
       %img = imshow( I1, [] );
       I2 = flip(I1 ,2);  %# horizontal flip
       I3 = flip(I1 ,1);
       I4 = flip(I3, 2);
       %figure(4)
       %img = imshow( I4, [] );%# horizontal+vertical flip
       %set(gcf,'position',[0,10,350,350])
       M1 = R1*images(i_label).M3x4;
       M2 = R2*M1; M3 = R3*M1;M4 = R4*M1;
       %R is rotation matrix for flipping
       img_data(:,:,data_count,i_label) = I1;
       cam_param(:,:,data_count,i_label) = M1;
       img_data(:,:,data_count+1,i_label) = I2;
       cam_param(:,:,data_count+1,i_label) = M2;
       img_data(:,:,data_count+2,i_label) = I3;
       cam_param(:,:,data_count+2,i_label) = M3;
       img_data(:,:,data_count+3,i_label) = I4;
       cam_param(:,:,data_count+3,i_label) = M4;
       Ms1 = M_lable(Ms1,images,R1,i_label);
       Ms2 = M_lable(Ms2,images,R2,i_label);
       Ms3 = M_lable(Ms3,images,R3,i_label);
       Ms4 = M_lable(Ms4,images,R4,i_label);
       
    end
    [lfr1,poly1] = Labels(site,thermalpath,thermalParams,labels,images,Ms1,1,K);
    [lfr2,poly2] = Labels(site,thermalpath,thermalParams,labels,images,Ms2,2,K);
    [lfr3,poly3] = Labels(site,thermalpath,thermalParams,labels,images,Ms3,3,K);
    [lfr4,poly4] = Labels(site,thermalpath,thermalParams,labels,images,Ms4,4,K);
    lfr(:,:,data_count) = lfr1;
    rect(data_count).pos = poly1;
    lfr(:,:,data_count+1) = lfr2;
    rect(data_count+1).pos = poly2;
    lfr(:,:,data_count+2) = lfr3;
    rect(data_count+2).pos = poly3;
    lfr(:,:,data_count+3) = lfr4;
    rect(data_count+3).pos = poly4;
    F_count=F_count+1;

   % Labels(site,thermalpath,thermalParams,images,Ms1,1,K);
end
data(i_site).lblimage = lfr;
data(i_site).lblpos = rect;
data(i_site).image = img_data;
data(i_site).cam_param = cam_param;
vars = {'rect','lfr','img_data','cam_param','Ms1','Ms2','Ms3','Ms4'};
clear(vars{:})
end
end

function Ms = M_lable(MS,images,R,i_label)
    Ms = MS;
    M = R*images(i_label).M3x4;
    M(4,:) = [0,0,0,1];
    Ms{i_label} = M;
end

function [lfr,POLY] = Labels(site,thermalpath,thermalParams,labels,images,Ms,flip_direction,K)

    refId = (round(length(images)/2))+1; % compute center by taking the average id!
    imgr = undistortImage( imread(fullfile(thermalpath,images(refId).imagefile)), ...
           thermalParams.cameraParams );
    imgr=double(imgr);
    imgr = imgr./max(max(imgr));
    imgr = imresize(imgr,[454 454]);
    
    if flip_direction == 1
        imgr = imgr;
    end
    if flip_direction == 2
           imgr = flip(imgr ,2);%Horizontal
    end
    if flip_direction == 3
           imgr = flip(imgr ,1);%Vertical
    end
    if flip_direction == 4
           imgr = flip(imgr ,2);
           imgr = flip(imgr ,1);%both
    end
        
    M1 = Ms{refId};
    R1 = M1(1:3,1:3)';
    t1 = M1(1:3,4)';
    range = [min(imgr(:)), max(imgr(:))];
    integral = zeros(size(imgr),'double');
    count = zeros(size(imgr),'double');

    for i_label = 1:length(images)
        img2 = undistortImage( imread(fullfile(thermalpath,images(i_label).imagefile)), ...
               thermalParams.cameraParams );
        img2=double(img2);
        img2 = img2./max(max(img2));
        img2 = imresize(img2,[454 454]);
        if flip_direction == 1
            img2 = img2;
        end
        if flip_direction == 2
           img2 = flip(img2 ,2);%Horizontal
        end
        if flip_direction == 3
           img2 = flip(img2 ,1);%Vertical
        end
        if flip_direction == 4
           img2 = flip(img2 ,2);
           img2 = flip(img2 ,1);%both
        end

        M2 = Ms{i_label};
        R2 = M2(1:3,1:3)';
        t2 = M2(1:3,4)';

        % relative 
        R = R1' * R2;
        t = t2 - t1 * R;
        z = getAGL( site ); % meters
        % the checkerboard is ~900 millimeters away
        % the tree in the background is ~100000 millimeters (100 m)
        P = (inv(K) * R * K ); 
        P_transl =  (t * K);
        P_ = P; % copy
        P_(3,:) = P_(3,:) + P_transl./z; % add translation
        tform = projective2d( P_ );

        % --- warp images ---
        % warp onto reference image
        warped2 = double(imwarp(img2,tform.invert(), 'OutputView',imref2d(size(imgr))));
        warped2(warped2==0) = NaN; % border introduced by imwarp are replaced by nan

        count(~isnan(warped2)) = count(~isnan(warped2)) + 1;
        integral(~isnan(warped2)) = integral(~isnan(warped2)) + warped2(~isnan(warped2));

    end
    lfr = integral ./ count;
    %h_fig = figure(100+linenumber); clf; % continue figure ...
    %set( h_fig, 'name', sprintf( '%s line %d', site, linenumber ) );
    %figure(2)
    %imshow( lfr, [] );
    %lbl(data_count).image = lfr;


    % project labels
    %figure(2);
    K_ = K; K_(4,4) = 1.0; % make sure intrinsic is 4x4

    % draw polygon
    
    for i_label = 1:length(labels)
       if isempty(labels) || isempty(labels(i_label).poly)
           continue;
       end
       
       x = labels(i_label).poly(:,1);
       y = labels(i_label).poly(:,2);
       if all(x)==0 && all(y)==0
           continue
       end
       polyin = polyshape({x},{y});
       [pol_x,pol_y] = boundingbox(polyin,1);
       poly_x = [pol_x(1);pol_x(1);pol_x(2);pol_x(2)];
       poly_y = [pol_y(1);pol_y(2);pol_y(2);pol_y(1)];
       
       
       %change the scale of label position
       moved_POS = [(454/640).*poly_x,... 
                    (454/512).*poly_y];
       if flip_direction == 1
           poly = moved_POS;
       end
       if flip_direction == 2
           new_pos = (454/2)*[1;1;1;1 ]-moved_POS(:,1);
           poly = [(454/2)*[1;1;1;1]+new_pos,moved_POS(:,2)];
       end
       if flip_direction == 3
           new_pos = (454/2)*[1;1;1;1 ]-moved_POS(:,2);
           poly = [moved_POS(:,1),(454/2)*[1;1;1;1]+new_pos];
       end
       if flip_direction == 4
           new_pos = (454/2)*[1 1;1 1;1 1;1 1]-moved_POS;
           poly = (454/2)*[1 1;1 1;1 1;1 1]+new_pos;
       end
       %lbl(data_count).poly = poly;    
       %poly = labels(i_label).poly;
       POLY(:,:,i_label)= poly;
       assert( strcmpi( images(refId).imagefile, labels(i_label).imagefile ), 'something went wrong: imagefile names of label and poses do not match!' );
       drawpolygon( 'Position', poly );
    end

end