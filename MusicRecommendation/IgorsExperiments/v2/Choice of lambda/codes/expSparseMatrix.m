function Mexp = expSparseMatrix(M, I)
    [pairsI, pairsJ, pairsV] = find(I);
    T = length(pairsI);
    
    for k=1:T
        r = M(pairsI(k), pairsJ(k));
        pairsV(k) = exp(r);
    end
    
    Mexp = sparse(pairsI, pairsJ, pairsV, size(M, 1), size(M, 2));
end

