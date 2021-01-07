function nmsProposalStruct = nonMaximumSupression(proposalStruct)
%Implementation of Non Maximum Supression
%   Non Maximum Supression using the intersection over union
%Input: A struct (proposalStruct) with a 2 substuct:
%   list of Proposal bounding boxes
%   list of confidences corresponding to the bbs
%Output: A stuct of filtered BoundingBoxes
%Link to resource:
%https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c


%% Preprocessing
%struct with bbs and confidences where the confidence is above a chosen
%threshold
filterThreshold = 0.1;

[lenPropStruct, ~] = size(proposalStruct.bbs);
filteredStruct.bbs = [];
filteredStruct.confidences = [];
filteredCounter = 1; 

for i=1 : lenPropStruct
    bi = proposalStruct.bbs(i,:);
    ci = proposalStruct.confidences(i);
    
    if ci > filterThreshold
        filteredStruct.bbs(filteredCounter,:) = bi;
        filteredStruct.confidences(filteredCounter) = ci;
        filteredCounter = filteredCounter + 1;
    end
end

%% The actual NMS
threshold = 0.5;
nmsProposalStruct.bbs = [];
nmsProposalStruct.confidences = [];
nmsProposalCounter = 1;
[lenFilteredStruct, ~] = size(filteredStruct.bbs);

for i=1 : lenFilteredStruct    
    discard = false;
    bi = filteredStruct.bbs(i,:);
    ci = filteredStruct.confidences(i);
    
    for j = 1 : lenFilteredStruct
        bj = filteredStruct.bbs(j,:);
        cj = filteredStruct.confidences(j);
        %calculate Intersecion Over Union
    iou = bboxOverlapRatio(bi, bj, 'Union');
        if iou > threshold
            if cj > ci
                discard = true;
            end
        end

    end
    if ~discard
        %values are stored horizontally in 2D-Array 
        nmsProposalStruct.bbs(nmsProposalCounter,:) = bi;
        nmsProposalStruct.confidences(nmsProposalCounter) = ci;
        nmsProposalCounter = nmsProposalCounter +1;
    end
end
end

