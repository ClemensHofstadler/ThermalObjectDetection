function fusedBbs = bbsFusion(proposalStruct)

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
fusedBbs.bbs = [];
fusedBbs.confidences = [];
[lenFilteredStruct, ~] = size(filteredStruct.bbs);

for i=1 : lenFilteredStruct    
    bi = filteredStruct.bbs(i,:);
    ci = filteredStruct.confidences(i);
    currentBbs = bi;
    for j = 1 : lenFilteredStruct
        
        bj = filteredStruct.bbs(j,:);
        cj = filteredStruct.confidences(j);
        %calculate Intersecion Over Union
        
        if ~isequal(currentBbs, bj)
            
            iou = bboxOverlapRatio(currentBbs, bj, 'Union');
            if iou > 0 
                xmin = min(bi(1),bj(1));
                xmax = max(bi(1) + bi(3) ,bj(1) + bj(3));
                ymin = min(bi(2),bj(2));
                ymax = max(bi(2) + bi(4), bj(2) + bj(4));

                maxWidth = xmax - xmin;
                maxHeight = ymax - ymin;
                
%                 if maxHeight == 0
%                     maxHeight = bi(4);
%                 else
%                     maxHeight = bi(4) + bj(4) - maxHeight;
%                 end
%                     
%                 if maxWidth == 0
%                     maxWidth = bi(3);
%                 else
%                     maxWidth = bi(3) + bj(3) - maxWidth;
%                 end
                
                currentBbs = [xmin, ymin, maxWidth, maxHeight];
                currentConf = max(ci,cj);
            end
        end
    end
    fusedBbs.bbs(i,:) = currentBbs;
    fusedBbs.confidences(i,:) = currentConf;
end
end