function [predictionsTr, predictionsTe, rmse1, rmse2, rmse3] = getBaseLinePredictions(trainTrueR, testTrueR, weight)
    
    [pairsI1, pairsJ1, pairsV1] = find(trainTrueR);
    trainT = length(pairsI1);
    [pairsI2, pairsJ2, pairsV2] = find(testTrueR);
    testT = length(pairsI2);

    numOfUsers = size(trainTrueR, 1);
    numOfItems = size(testTrueR, 2);
    
    U = zeros(numOfUsers, 1);
    for i=1:numOfUsers
        if (length(find(trainTrueR(i, :) > 0)) > 0)
            U(i) = mean(trainTrueR(i, find(trainTrueR(i, :) > 0)));
        end
    end

    M = zeros(numOfItems, 1);
    for j=1:numOfItems
        if (length(find(trainTrueR(:, j) > 0)) > 0)
            M(j) = mean(trainTrueR(find(trainTrueR(:, j) > 0), j));
        end
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
    
    predictions = zeros(testT, 1);
    % let's calculate RMSE
    rmse = 0;
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        r = pairsV2(k);
        prediction = weight * U(i) + (1 - weight) * M(j);
        rmse = rmse + (prediction - r) * (prediction - r);
        predictions(k) = prediction;
    end
    rmse3 = sqrt(rmse/testT);   
    predictionsTe = sparse(pairsI2, pairsJ2, predictions, numOfUsers, numOfItems);
    
    predictions = zeros(trainT, 1);
    for k=1:trainT
        i = pairsI1(k);
        j = pairsJ1(k);
        prediction = weight * U(i) + (1 - weight) * M(j);
        predictions(k) = prediction;
    end
    predictionsTr = sparse(pairsI1, pairsJ1, predictions, numOfUsers, numOfItems);
    
end
