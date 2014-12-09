function Mlog = logaritmizeSparseMatrix(M)
    [pairsI, pairsJ, pairsV] = find(M);
    T = length(pairsI);
    
    for k=1:T
        r = pairsV(k);
        pairsV(k) = log(r);
    end
    
    Mlog = sparse(pairsI, pairsJ, pairsV, size(M, 1), size(M, 2));
end

