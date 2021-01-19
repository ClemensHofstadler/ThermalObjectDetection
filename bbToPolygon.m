function coords = bbToPolygon(bb)
    coords = zeros(4,2);
    coords(1,:) = bb(1:2);
    coords(2,:) = bb(1:2) + [bb(3) 0];
    coords(3,:) = bb(1:2) + [bb(3) bb(4)];
    coords(4,:) = bb(1:2) + [0 bb(4)];
end