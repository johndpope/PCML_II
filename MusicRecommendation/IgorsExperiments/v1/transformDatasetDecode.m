function Y = transformDatasetDecode(Ytransformed, U, M, I)

    numOfUsers = size(Ytransformed, 1);
    numOfItems = size(Ytransformed, 2);
    
    [pairsI, pairsJ, pairsV] = find(I);
    T = length(pairsI);
    
    % let's calculate RMSE
    for k=1:T
        i = pairsI(k);
        j = pairsJ(k);
        r = Ytransformed(i, j);
        weight = (U(i) + M(j)) / 2;
        pairsV(k) = r + weight;
    end
    
    Y = sparse(pairsI, pairsJ, pairsV, numOfUsers, numOfItems);
    
end
