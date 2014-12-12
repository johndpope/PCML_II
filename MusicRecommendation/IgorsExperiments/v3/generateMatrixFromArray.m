function M = generateMatrixFromArray(arr, I)
    [pi, pj, pr] = find(I);
    M = sparse(pi, pj, arr, size(I, 1), size(I, 2));
end

