clear all; clc; close all; % clean up!
data_root_folder = 'D:\DATA\Studium JKU\2020W\Computer Vision UE\lab_03';
%%
addpath '.\util';
scene_filter = 'F8';

%% run
scene_struct = loadData(data_root_folder, scene_filter); 

%% Loads all scenes and creates a structure
% seq field of scene_struct is empty if no people are detected, ie. no

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
        drift = -getDrift(scene_struct(i_scene).name);
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
                    i = 1;
                    for i_rel = 1:length(rel_pos_struct)
                        R = R_lab' * rel_pos_struct(i_rel).M3x4(1:3,1:3)';
                        t = rel_pos_struct(i_rel).M3x4(1:3,4)' - t_lab * R;
                        rel_pos_struct(i_rel).M3x4(1:3,1:3) = R';
                        rel_pos_struct(i_rel).M3x4(1:3,4) = t';
                        i = i + 1;
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
                        if (i_seq == 13 && i_lab == 1)
                            i_lab = i_lab;
                        end
                        
                        M = rel_pos_struct(i_lab).M3x4;
                        % M(4,:) = [0, 0, 0, 1];
                        R = rel_pos_struct(i_lab).M3x4(1:3,1:3)';
                        t = rel_pos_struct(i_lab).M3x4(1:3,4)';
                        t(1) = t(1) + drift * (i_label - i);
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


%% 
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

%%
function s = reduceFolderStructure(s)
    s = rmfield(s, 'date');
    s = rmfield(s, 'bytes');
    s = rmfield(s, 'isdir');
    s = rmfield(s, 'datenum');
end








