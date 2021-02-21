function candidates = prediction(net, x, gridSize, inputSize)
    out = net.predict(x);
    candidates.confidences = out(:);
    candidates.bbs = [];
    idx = 1;
    for s = gridSize
       boxes_per_line = ceil(inputSize(1)/s);
       idx = idx + boxes_per_line^2;
       bbs = zeros([boxes_per_line boxes_per_line 4]);
       for i = 1:boxes_per_line
           for j = 1:boxes_per_line
               bbs(i,j,:) = [(j-1)*s+1 (i-1)*s+1 s+1 s+1];
           end
       end
       candidates.bbs = [candidates.bbs
           reshape(bbs,[boxes_per_line^2 4])];
    end
end