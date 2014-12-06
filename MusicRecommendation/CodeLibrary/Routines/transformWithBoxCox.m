function [Mout, lambda] = transformWithBoxCox(Min)
    [pairsI, pairsJ, pairsV] = find(Min);
    [transdat, lambda] = boxcox(pairsV);
    Mout = sparse(pairsI, pairsJ, transdat, size(Min, 1), size(Min, 2));
end
