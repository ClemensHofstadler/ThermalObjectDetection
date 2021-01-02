% FIRST: define the hyperparameters by running the hyperparameters.m script
% THEN: load/setup the net by running the model.m script
% THEN: rund this script to set up everything for training
% THEN: run trained_net = trainNetwork(X,Y,net,options); to actually train
% the network

% these parameters you can play around with
maxEpochs = 1;
miniBatchSize  = 32;
optimizer = 'adam';
initialLearnRate = 1e-3;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 20;
shuffleFrequency = 'every-epoch';
executionEnvironment = 'auto';

[X,Y] = prepareData(data_root_folder, scene_filter, inputSize, gridSize);

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
    labelGrid = [];
    for s = gridSize
        boxes_per_line = ceil(inputSize(1)/s);
        currentGrid = zeros(boxes_per_line,boxes_per_line);
        for i_label = 1:size(label,3)
            x_min = min(label(:,1,i_label));
            x_max = max(label(:,1,i_label));
            y_min = min(label(:,2,i_label));
            y_max = max(label(:,2,i_label));
            for i = 1:boxes_per_line
                for j = 1:boxes_per_line
                    left_right_inside_box = floor(x_min/s) <= i && i <= ceil(x_max/s);
                    up_down_inside_box = floor(y_min/s) <= j && j <= ceil(y_max/s);  
                    if left_right_inside_box && up_down_inside_box
                        currentGrid(i,j) = 1;
                    end
                end
            end
        end
        labelGrid = [labelGrid
            currentGrid(:)];
    end
end

function [X,Y] = prepareData(data_root_folder, scene_filter, inputSize, gridSize)
    fprintf('Loading data...\n');
    folder = loadData(data_root_folder, scene_filter);
    fprintf('Augmenting data...\n');  
    data = dta_loader(folder);
    fprintf('Preparing data for training...\n');  
    [X,Y] = prepDataAux(data,inputSize,gridSize);
end

function [X,Y] = prepDataAux(data, inputSize, gridSize)
    num_seqs = 0;
    for i = 1:length(data)
        num_seqs = num_seqs + size(data(i).seq,2);
    end
    X = cell(num_seqs,1);
    Y = cell(num_seqs,1);
    idx = 1;
    numOutputs = 0;
    for s = gridSize
        numOutputs = numOutputs + ceil(inputSize(1)/s)^2; 
    end
    for i = 1:length(data)
        current_folder = data(i).seq;
        X_from_folder = cell(length(current_folder),1);
        Y_from_folder = cell(length(current_folder),1);
        for j = 1:length(current_folder)
        	current_seq = current_folder(j);
            seq_len = size(current_seq.image,2);
            current_X = zeros([inputSize seq_len]);
            current_Y = zeros([numOutputs seq_len]);  
            % save images in first dimension and relative poses in second
            % dimension
            current_X(:,:,1,:) = permute(current_seq.image,[3 4 2 1]);
            current_X(1:3,1:4,2,:) = permute(current_seq.cam_param,[3 4 2 1]);
            % generate label grid and save it
            for k = 1:seq_len
                if ~isempty(current_seq.poly{k})
                    current_Y(:,k) = generateLabelGrid(current_seq.poly{k},inputSize,gridSize);
                end
            end
            X_from_folder{j} = single(current_X);
            Y_from_folder{j} = single(current_Y);
        end
        X(idx:idx+length(current_folder)-1) = X_from_folder;
        Y(idx:idx+length(current_folder)-1) = Y_from_folder;
        idx = idx + length(current_folder);
    end
end

function sceneStruct = loadData(data_root_folder, scene_filter)
    folder = fullfile(data_root_folder, "data");
    params = load(fullfile(folder, 'camParams_thermal.mat'));
    K = params.cameraParams.Intrinsics.IntrinsicMatrix;
    scene_struct = LoadStructureFolderInFolderFiltered(folder, scene_filter);
    scene_struct = reduceFolderStructure(scene_struct);

    % add sequence-structure field to scene
    scene_struct(end).seq = [];

    % add sequence-structures
    for i_scene = 1:length(scene_struct)
        % getting the z for the entire scene
        z = getAGL(scene_struct(i_scene).name);
        try
            folder = fullfile(scene_struct(i_scene).folder, scene_struct(i_scene).name, 'Images');
            seq_struct = LoadStructureFolderInFolderFiltered(folder, '');
            seq_struct = reduceFolderStructure(seq_struct);
            % add fields for image, label, pose, and relative pose arrays 
            seq_struct(end).images = [];
            seq_struct(end).labels_rect = [];
            seq_struct(end).poses = [];
            seq_struct(end).relative_poses = [];

            for i_seq = 1:length(seq_struct)
                
                % images ----------
                folder_seq = fullfile(seq_struct(i_seq).folder, seq_struct(i_seq).name, '*.tiff');
                img_struct = dir(folder_seq);
                img_struct = reduceFolderStructure(img_struct);
                img_struct(end).data = [];
                for i_img = 1:length(img_struct)
                    thermalds = datastore(fullfile(img_struct(i_img).folder, img_struct(i_img).name));
                    img_struct(i_img).data = readimage(thermalds, 1);
                end
                
                % mid label ----------
                integral = zeros(size(img_struct(1).data),'double');
                x_max = length(integral(1,:));
                y_max = length(integral(:,1));
                label_file_path = fullfile(scene_struct(i_scene).folder, scene_struct(i_scene).name, 'Labels', append('Label', seq_struct(i_seq).name, '.json'));
                label_mid_struct = jsondecode(fileread(label_file_path));
                label_mid_struct = label_mid_struct.Labels;
                % make labels rectangle
                for i_lab = 1:length(label_mid_struct)
                    [absBBs, relBBs, ~] = saveLabels({label_mid_struct(i_lab).poly}, size(integral), []);
                    x1 = absBBs(1,1); 
                    x2 = absBBs(1,2); 
                    y1 = absBBs(1,3); 
                    y2 = absBBs(1,4);
                    label_mid_struct(i_lab).poly = [[x2 y1];[x1 y1];[x1 y2];[x2 y2]];
                end
                
                % poses -----------
                poses_file_path = fullfile(scene_struct(i_scene).folder, scene_struct(i_scene).name, 'Poses', append(seq_struct(i_seq).name, '.json'));
                pos_struct = jsondecode(fileread(poses_file_path));
                pos_struct = pos_struct.images;
                % calculate relative poses
                rel_pos_struct = pos_struct;
                try
                    try 
                        i_label = find(ismember({pos_struct.imagefile}, label_mid_struct(1).imagefile));
                    catch ee
                        if mod(length(pos_struct), 2) == 0
                            i_label = int8(length(pos_struct) / 2.0);
                        else
                            i_label = 1 + int8(length(pos_struct) / 2.0);
                        end
                    end
                    R_lab = pos_struct(i_label).M3x4(1:3,1:3)';
                    t_lab = pos_struct(i_label).M3x4(1:3,4)';
                    for i_rel = 1:length(rel_pos_struct)
                        R = R_lab' * rel_pos_struct(i_rel).M3x4(1:3,1:3)';
                        t = rel_pos_struct(i_rel).M3x4(1:3,4)' - t_lab * R;
                        rel_pos_struct(i_rel).M3x4(1:3,1:3) = R';
                        rel_pos_struct(i_rel).M3x4(1:3,4) = t';
                    end
                catch e
                    rel_pos_struct = [];
                    fprintf('%s\n\r', append('No labels (i.e.no people) in scene ', scene_struct(i_scene).name, ", in sequence ", seq_struct(i_seq).name));  
                    fprintf('%s\n\r', append('error message: ', e.message));
                end

                % labels ----------
                lab_struct = pos_struct;
                lab_struct = rmfield(lab_struct, 'M3x4');
                lab_struct(end).labels = [];
                for i_lab = 1:length(lab_struct)
                    try
                        % only for testing
                        if (i_scene == 8 && i_seq == 5 && i_lab == 16)
                            i_lab = i_lab;
                        end
                        
                        M = rel_pos_struct(i_lab).M3x4;
                        % M(4,:) = [0, 0, 0, 1];
                        R = rel_pos_struct(i_lab).M3x4(1:3,1:3)';
                        t = rel_pos_struct(i_lab).M3x4(1:3,4)';
                        lab_struct(i_lab).labels = label_mid_struct;
                        % for each label bounding box in mid image label definition...
                        for i_bb = 1:length(label_mid_struct)
                            frame_4P = label_mid_struct(i_bb).poly;
                            frame_4P(:,end+1) = z;
                            % each point of frame...
                            for i_pnt = 1:length(frame_4P)
                                x_o = frame_4P(i_pnt,:);
                                x_d = (x_o * inv(K) * R + t) * K / z;
                                x_n = x_o + x_d;
                                x_n(:,end) = [];
                                % adapt label bounding boxes to image boarder
                                % x-value
                                if x_n(1) < 0  
                                    x_n(1) = 0;
                                else
                                    if x_n(1) > x_max 
                                        x_n(1) = x_max;
                                    end
                                end
                                % y-value
                                if x_n(2) < 0  
                                    x_n(2) = 0;
                                else
                                    if x_n(2) > y_max 
                                        x_n(2) = y_max;
                                    end
                                end
                                lab_struct(i_lab).labels(i_bb).poly(i_pnt,:) = x_n;
                            end
                            % check if bounding box has an extension
                            
                        end
                        % erase a bounding box with 0 area
                        for i_bb = 1:length(lab_struct(i_lab).labels)
                            if (lab_struct(i_lab).labels(i_bb).poly(1,1) == lab_struct(i_lab).labels(i_bb).poly(2,1) && ...
                                lab_struct(i_lab).labels(i_bb).poly(2,1) == lab_struct(i_lab).labels(i_bb).poly(3,1) && ...
                                lab_struct(i_lab).labels(i_bb).poly(3,1) == lab_struct(i_lab).labels(i_bb).poly(4,1)) || ...
                               (lab_struct(i_lab).labels(i_bb).poly(1,2) == lab_struct(i_lab).labels(i_bb).poly(2,2) && ...
                                lab_struct(i_lab).labels(i_bb).poly(2,2) == lab_struct(i_lab).labels(i_bb).poly(3,2) && ...
                                lab_struct(i_lab).labels(i_bb).poly(3,2) == lab_struct(i_lab).labels(i_bb).poly(4,2))
                                lab_struct(i_lab)
                                lab_struct(i_lab).labels(i_bb) = [];
                            end
                        end
                    catch e
                        fprintf('%s\n\r', append('error message: ', e.message));
                    end
                end
                
                seq_struct(i_seq).relative_poses = rel_pos_struct;
                seq_struct(i_seq).poses = pos_struct;
                seq_struct(i_seq).labels_rect = lab_struct;
                seq_struct(i_seq).images = img_struct;
            end

            scene_struct(i_scene).seq = seq_struct;
        catch e
            fprintf('%s\n\r', append('error in scene ', scene_struct(i_scene).name));
            fprintf('%s\n\r', append('error message: ', e.message));
        end
    end
    sceneStruct = scene_struct;
end

function folderStructFiltered = LoadStructureFolderInFolderFiltered(folder, filterStringPefix)
    % load scene structure
    mainStruct = dir(folder);
    mainStruct(~[mainStruct.isdir]) = [];   % remove non dirs
    toRemove = ismember( {mainStruct.name}, {'.', '..'});
    mainStruct(toRemove) = [];  %remove current and parent directory
    toRemove = false(1, length(mainStruct));
    filteredList = regexp({mainStruct.name}, filterStringPefix + "\w*", 'match');
    for i = 1:length(toRemove)
        if isempty(filteredList{i})
            toRemove(i) = 1;
        else
            toRemove(i) = 0;
        end
    end
    mainStruct(toRemove) = [];  %remove all directories that do not match the filterStringPefix criteria
    
    folderStructFiltered = mainStruct;
end

function s = reduceFolderStructure(s)
    s = rmfield(s, 'date');
    s = rmfield(s, 'bytes');
    s = rmfield(s, 'isdir');
    s = rmfield(s, 'datenum');
end

function [data] = dta_loader(Folder)
thermalParams = load( './data/camParams_thermal.mat' );
K = thermalParams.cameraParams.IntrinsicMatrix;
cam_param = thermalParams.cameraParams;
R1 =[1 0 0;0 1 0;0 0 1];
R2 =[-1 0 0;0 1 0;0 0 1];
R3 =[1 0 0;0 -1 0;0 0 1];
R4 =[-1 0 0;0 -1 0;0 0 1];
Rs = [454/640 0 0;0 454/512 0;0 0 1]; %scaling images causes Intrinsic changes
resize_shape = [454 454];

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
    seqs = zeros([4 length(images) resize_shape]);
    cam_params = zeros([4 length(images) 3 4]);
    line_labels = cell(4,length(images));
    
    for i_label = 1:length(images)
       thermal = undistortImage(images(i_label).data,cam_param);
       thermal = double(thermal);
       thermal = thermal./max(max(thermal));
       lbl = labels(i_label).labels;
       
       %Normalize Pics
       I1 = imresize(thermal,resize_shape);     %resize pics
       %figure(1)
       %img = imshow( I1, [] );
       I2 = flip(I1 ,2);  %# horizontal flip
       I3 = flip(I1 ,1);
       I4 = flip(I3, 2);
       %figure(4)
       %img = imshow( I4, [] );%# horizontal+vertical flip
       %set(gcf,'position',[0,10,350,350])
       %i_site
       %linenumber
       %i_label
       M1 = R1*poses(i_label).M3x4;
       M2 = R2*M1; M3 = R3*M1;M4 = R4*M1;
       %R is rotation matrix for flipping
       
       seqs(1,i_label,1:end,1:end) = I1;
       seqs(2,i_label,1:end,1:end) = I2;
       seqs(3,i_label,1:end,1:end) = I3;
       seqs(4,i_label,1:end,1:end) = I4;
       
       cam_params(1,i_label,1:end,1:end) = M1;
       cam_params(2,i_label,1:end,1:end) = M2;
       cam_params(3,i_label,1:end,1:end) = M3;
       cam_params(4,i_label,1:end,1:end) = M4;
       
       line_labels{1,i_label} = single(Labels(I1,lbl,1,K));
       line_labels{2,i_label} = single(Labels(I2,lbl,2,K));
       line_labels{3,i_label} = single(Labels(I3,lbl,3,K));
       line_labels{4,i_label} = single(Labels(I4,lbl,4,K));
      
    end
    data(i_site).seq(data_count).image = single(seqs(1,1:end,1:end,1:end));
    data(i_site).seq(data_count+1).image = single(seqs(2,1:end,1:end,1:end));
    data(i_site).seq(data_count+2).image = single(seqs(3,1:end,1:end,1:end));
    data(i_site).seq(data_count+3).image = single(seqs(4,1:end,1:end,1:end));
    
    data(i_site).seq(data_count).cam_param = single(cam_params(1,1:end,1:end,1:end));
    data(i_site).seq(data_count+1).cam_param = single(cam_params(2,1:end,1:end,1:end));
    data(i_site).seq(data_count+2).cam_param = single(cam_params(3,1:end,1:end,1:end));
    data(i_site).seq(data_count+3).cam_param = single(cam_params(4,1:end,1:end,1:end));
    
    data(i_site).seq(data_count).poly = line_labels(1,:);
    data(i_site).seq(data_count+1).poly = line_labels(2,:);
    data(i_site).seq(data_count+2).poly = line_labels(3,:);
    data(i_site).seq(data_count+3).poly = line_labels(4,:);
    
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
    %figure(1)
    %imshow(image,[])
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
       %poly = label(i_label).poly;
       POLY(:,:,i_label) = poly;
       %drawpolygon( 'Position', poly );
       %title(num2str(i_label));
       %hold on
    end
    %hold off
    
    
end