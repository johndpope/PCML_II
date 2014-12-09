function [Ytransformed, I] = transformDataset2(Y, U, M)

    numOfUsers = size(Y, 1);
    numOfItems = size(Y, 2);
    
    I = Y;
    [pairsI, pairsJ, pairsV] = find(I);
    T = length(pairsI);
    for i=1:T
        pairsV(i) = 1;
    end
    I = sparse(pairsI, pairsJ, pairsV, numOfUsers, numOfItems);
    
    [pairsI, pairsJ, pairsV] = find(Y);
    T = length(pairsI);
    
    % let's calculate RMSE
    for k=1:T
        i = pairsI(k);
        j = pairsJ(k);
        r = pairsV(k);
        weight = (U(i) + M(j)) / 2;
        pairsV(k) = r - weight;
    end
    
    Ytransformed = sparse(pairsI, pairsJ, pairsV, numOfUsers, numOfItems);
    
end
