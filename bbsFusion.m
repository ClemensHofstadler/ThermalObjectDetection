function filtered = bbsFusion(proposalStruct, candidate_confidence)

%% Preprocessing
%struct with bbs and confidences where the confidence is above a chosen
%threshold
filterThreshold = candidate_confidence;

lenPropStruct = size(proposalStruct.bbs,1);
filtered.bbs = [];
filtered.confidences = [];

for i=1 : lenPropStruct
    if  proposalStruct.confidences(i) > filterThreshold
        filtered.bbs = [filtered.bbs
            proposalStruct.bbs(i,:)];
        filtered.confidences = [filtered.confidences
            proposalStruct.confidences(i)];
    end
end

%% The actual NMS
mergedBBs = true;
while mergedBBs
    current.bbs = [];
    current.conf = [];
    mergedBBs = false;
    while size(filtered.bbs,1) > 0
        % remove 1st bb
        bbi = filtered.bbs(1,:);
        ci = filtered.confidences(1);
        filtered.bbs(1,:) = [];
        filtered.confidences(1) = [];
        % compare 1st bb to all others
        mergedWithBBi = false;
        j = 1;
        while j <= size(filtered.bbs,1)
            bbj = filtered.bbs(j,:);
            if bboxOverlapRatio(bbi, bbj, 'Union') > 0 
                mergedBBs = true;
                mergedWithBBi = true;
                xmin = min(bbi(1),bbj(1));
                xmax = max(bbi(1) + bbi(3) ,bbj(1) + bbj(3));
                ymin = min(bbi(2),bbj(2));
                ymax = max(bbi(2) + bbi(4), bbj(2) + bbj(4));
                % add merged bb
                 current.bbs = [current.bbs
                    [xmin, ymin, xmax - xmin, ymax - ymin]];
                current.conf = [current.conf
                    max(ci,filtered.confidences(j))];
                 % remove jth bb
                 filtered.bbs(j,:) = [];
                 filtered.confidences(j) = [];
            else
                j = j+1;
            end
        end
        if ~mergedWithBBi
            current.bbs = [current.bbs
                   bbi];
            current.conf = [current.conf
                    ci];
        end   
    end
    filtered.bbs = current.bbs;
    filtered.confidences =  current.conf;
end

end