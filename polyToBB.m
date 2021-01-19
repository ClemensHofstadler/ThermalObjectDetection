function bbs = polyToBB(poly)
    bbs = zeros(size(poly,3),4);
    for i = 1:size(poly,3)
       xmin = min(poly(:,1,i));
       xmax = max(poly(:,1,i));
       ymin = min(poly(:,2,i));
       ymax = max(poly(:,2,i));
       bbs(i,:) = [xmin ymin xmax-xmin ymax-ymin];
    end
end