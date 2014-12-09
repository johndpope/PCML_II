function I = getI(Ytr)
    numOfUsers = size(Ytr, 1);
    numOfItems = size(Ytr, 2);
    I = Ytr;
    [pairsI, pairsJ, pairsV] = find(I);
    T = length(pairsI);
    for i=1:T
        pairsV(i) = 1;
    end
    I = sparse(pairsI, pairsJ, pairsV, numOfUsers, numOfItems);
end
