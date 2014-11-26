function [] = writeMatrix( Y, filename )
    idx = find(Y(:) > 0);
    [I, J] = ind2sub(size(Y), idx);
    dlmwrite(filename, [full(I) full(J) full(Y(idx))], ',');
end

