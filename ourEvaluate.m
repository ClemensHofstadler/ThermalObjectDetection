detections = bbsFusion(candidates);
imgsize = [227, 227];
[detectionsLength, ~] = size(detections.bbs);

gts = cell(detectionsLength,1);
convertedBBS = cell(detectionsLength,1);


for i=1 : detectionsLength
    bb = yoloToBB(imgsize, detections.bbs(i,:));
    gts{i} = [bb(1) ,bb(2), bb(3), bb(4)];
    convertedBBS{i} = [detections.bbs(i,1),detections.bbs(i,2),detections.bbs(i,3),detections.bbs(i,4)];
end

cellConfidences = num2cell(detections.confidences);
matGT = table( gts, 'VariableNames', {'person'} );
matDetections = table( convertedBBS, cellConfidences, 'VariableNames', {'bboxes', 'scores'} );
iou_threshold = .10;
conf_threshold = .09;


averagePrecision = evaluateDetectionPrecision(matDetections,matGT,iou_threshold);
%Average precision over all the detection results, returned as a numeric scalar or vector. 

[FP, TP, GT] = computeFpTpFn( matDetections, matGT, iou_threshold, conf_threshold );
% computes false and true positives, given detections, ground truths an iou and confidence threshold.