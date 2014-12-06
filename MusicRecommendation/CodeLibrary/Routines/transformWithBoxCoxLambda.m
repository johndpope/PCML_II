function Mout = transformWithBoxCoxLambda(lambda, Min)
    [pairsI, pairsJ, pairsV] = find(Min);
    transdatTe = boxcox(lambda, pairsV);
    Mout = sparse(pairsI, pairsJ, transdatTe, size(Min, 1), size(Min, 2));
end

