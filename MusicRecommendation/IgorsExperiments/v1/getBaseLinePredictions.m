function [rmse1, rmse2, rmse3] = getBaseLinePredictions(trainTrueR, testTrueR)
    
    [pairsI1, pairsJ1, pairsV1] = find(trainTrueR);
    trainT = length(pairsI1);
    [pairsI2, pairsJ2, pairsV2] = find(testTrueR);
    testT = length(pairsI2);

    numOfUsers = size(trainTrueR, 1);
    numOfItems = size(testTrueR, 2);
    
    U = zeros(numOfUsers, 1);
    for i=1:numOfUsers
        U(i) = mean(trainTrueR(i, find(trainTrueR(i, :) > 0)));
    end

    M = zeros(numOfItems, 1);
    for j=1:numOfItems
        M(j) = mean(trainTrueR(find(trainTrueR(:, j) > 0), j));
    end
    
    % let's calculate RMSE
    rmse = 0;
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        r = pairsV2(k);
        prediction = U(i);
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmse1 = sqrt(rmse/testT);
    
    % let's calculate RMSE
    rmse = 0;
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        r = pairsV2(k);
        prediction = M(j);
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmse2 = sqrt(rmse/testT);
    
    % let's calculate RMSE
    rmse = 0;
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        r = pairsV2(k);
        prediction = (U(i) + M(j)) / 2;
        % I need to try weighted summation
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmse3 = sqrt(rmse/testT);
    
end

