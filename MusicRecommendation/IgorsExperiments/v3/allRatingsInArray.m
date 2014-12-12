function vals = allRatingsInArray(M, I)
    [pairsI, pairsJ, vals] = find(I);
    T = length(pairsI);
    for k=1:T
        vals(k) = M(pairsI(k), pairsJ(k));
    end
end

