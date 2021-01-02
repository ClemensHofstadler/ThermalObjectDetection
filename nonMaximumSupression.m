function nmsProposalStruct = nonMaximumSupression(proposalStruct)
%Implementation of Non Maximum Supression
%   Non Maximum Supression using the intersection over union
%Input: A struct (proposalStruct) with a 2 substuct:
%   list of Proposal bounding boxes
%   list of confidences corresponding to the bbs
%Output: A stuct of filtered BoundingBoxes
%Link to resource:
%https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c

threshold = 0.5;
nmsProposalStruct.bbs = [];
nmsProposalStruct.confidences = [];
nmsProposalCounter = 1;
[lenBbs, ~] = size(proposalStruct.bbs);

for i=1 : lenBbs    
    discard = false;
    bi = proposalStruct.bbs(i,:);
    ci = proposalStruct.confidences(i);
    
    for j = 1 : lenBbs
        bj = proposalStruct.bbs(j,:);
        cj = proposalStruct.confidences(j);
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

