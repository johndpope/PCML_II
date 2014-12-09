function r = spdiag(d)
    
    idx = find(d > 0);
    L = length(idx);
    pairsI = zeros(L, 1);
    pairsJ = zeros(L, 1);
    pairsV = zeros(L, 1);
    for i=1:L
        pairsI(i) = idx(i);
        pairsJ(i) = idx(i);
        pairsV(i) = d(idx(i));
    end
    r = sparse(pairsI, pairsJ, pairsV, length(d), length(d));
end
