function [Ytr, Yte] = getSplitDataWeak(Y, G, seed, numD)

    % numD is the number of artists held out per user (default 10)

    % Test data for Strong generalization
    % keep 10% of users for testing as 'new users'
    % You should decide on your own how many new users you want to test on
    setSeed(seed);
    
    % Test data for weak generalization
    % Keep 10 entries per existing user as test data
    [numOfUsers, numOfItems] = size(Y);
    pairsI = [];
    pairsJ = [];
    pairsV = [];
    for j=1:numOfItems
        On = find(Y(:, j) ~= 0);
        
        % if item j is listened by more than 2 users
        if length(On) > 2
            
            % what if numD > length(On), there will be duplicates also
            
            ind = unidrnd(length(On), numD, 1); % choose some for testing
            d = On(ind);
            pairsI = [pairsI; d];
            pairsJ = [pairsJ; j * ones(numD, 1)];
            pairsV = [pairsV; Y(d, j)];
        end
    end
    
    Yte = sparse(pairsI, pairsJ, pairsV, numOfUsers, numOfItems);
    Ytr = Y;
    Ytr(sub2ind([numOfUsers numOfItems], pairsI, pairsJ)) = 0;
    
end
