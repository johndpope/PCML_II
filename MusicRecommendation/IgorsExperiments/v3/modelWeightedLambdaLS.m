function [predictionsTr, predictionsTe, rmseStart, rmseEnd, rmseBest, U, M] = modelWeightedLambdaLS(Ytr, Yte, Itr, Ite, weight, G, seed, numOfLatentFactors, lambdaU, lambdaM, lambdaT, friendshipIncluded, numOfIterations)

    setSeed(seed);
    numOfUsers = size(Ytr, 1);
    numOfItems = size(Ytr, 2);

    % we vectorize the train data set
    [pairsI1, pairsJ1, pairsV1] = find(Itr);
    trainT = length(pairsI1);
    for k=1:trainT
        pairsV1(k) = Ytr(pairsI1(k), pairsJ1(k));
    end
    
    % we vectorize the test data set
    [pairsI2, pairsJ2, pairsV2] = find(Ite);
    testT = length(pairsI2);
    for k=1:testT
        pairsV2(k) = Yte(pairsI2(k), pairsJ2(k));
    end
    
    % we find the number of friends for each user
    numOfFriends = zeros(numOfUsers, 1);
    for i=1:numOfUsers
        numOfFriends(i) = sum(G(i, :));
    end

    % we initialize our solution by random
    U = randn(numOfLatentFactors, numOfUsers);
    M = randn(numOfLatentFactors, numOfItems);

    numOfRatingsPerUser = zeros(numOfUsers, 1);
    numOfRatingsPerItem = zeros(numOfItems, 1);

    for i=1:numOfUsers
        numOfRatingsPerUser(i) = length(find(Itr(i, :) == 1));
    end

    for j=1:numOfItems
        numOfRatingsPerItem(j) = length(find(Itr(:, j) == 1));
    end

    % initial RMSE for the train set
    rmse = 0.0;
    for k=1:trainT
        i = pairsI1(k);
        j = pairsJ1(k);
        r = pairsV1(k);
        prediction = U(:, i)' * M(:, j);
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmseTr = sqrt(rmse/trainT);

    % initial RMSE for the test set
    rmse = 0.0;
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        r = pairsV2(k);
        prediction =  U(:, i)' * M(:, j);
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmseTe = sqrt(rmse/testT);
    fprintf('Iteration %d. Train: %f, test: %f.\n', 0, rmseTr, rmseTe);
    
    rmseStart = rmseTe;
    
    rmseTrVector = zeros(numOfIterations, 1);
    rmseTeVector = zeros(numOfIterations, 1);

    % precalculations
    splf = speye(numOfLatentFactors);
    onesnu = ones(numOfUsers, 1);
    
    for iteration=1:numOfIterations
        
        newU = zeros(numOfLatentFactors, numOfUsers);
        newM = zeros(numOfLatentFactors, numOfItems);
        
        % for each user
        for i=1:numOfUsers
            
            if (numOfRatingsPerUser(i) > 0)
                Ri = Ytr(i, :);
                D = spdiag(weight(i, :));
                tmp1 = M * D * M';
                tmp2 = lambdaU * numOfRatingsPerUser(i) * splf;
                tmp3 = tmp1 + tmp2;
                tmp4 = M * D * Ri';
                
                if ((friendshipIncluded == 1)&&(numOfFriends(i) > 0))
                    tmp4 = tmp4 + (lambdaT / numOfFriends(i)) * U * spdiag(G(i, :)) * onesnu;
                end
                
                if (rank(tmp3) == numOfLatentFactors)
                    newU(:, i) = tmp3 \ tmp4;
                else
                    newU(:, i) = U(:, i);
                end
            end
            
        end
        
        U = newU;

        % for each item
        for j=1:numOfItems
            
            if (numOfRatingsPerItem(j) > 0)
                Rj = Ytr(:, j);
                D = spdiag(weight(:, j));
                tmp1 = U * D * U';
                tmp2 = lambdaM * numOfRatingsPerItem(j) * splf;
                tmp3 = tmp1 + tmp2;
                tmp4 = U * D * Rj;
                if (rank(tmp3) == numOfLatentFactors)
                    newM(:, j) = tmp3 \ tmp4;
                else
                    newM(:, j) = M(:, j);
                end
            end
            
        end

        M = newM;
    
        % let's calculate RMSE
        rmse = 0;
        for k=1:trainT
            i = pairsI1(k);
            j = pairsJ1(k);
            r = pairsV1(k);
            prediction = U(:, i)' * M(:, j);
            rmse = rmse + (prediction - r) * (prediction - r);
        end
        rmse = sqrt(rmse/trainT);
        rmseTrVector(iteration) = rmse;

        rmse = 0;
        for k=1:testT
            i = pairsI2(k);
            j = pairsJ2(k);
            r = pairsV2(k);
            prediction = U(:, i)' * M(:, j);
            rmse = rmse + (prediction - r) * (prediction - r);
        end
        rmse = sqrt(rmse/testT);
        rmseTeVector(iteration) = rmse;

        fprintf('Iteration %d. Train: %f, test: %f.\n', iteration, rmseTrVector(iteration), rmseTeVector(iteration));
    
    end
    
    rmseEnd = rmseTeVector(numOfIterations);
    rmseBest = rmseStart;
    for i=1:numOfIterations
        if (rmseTeVector(i) < rmseBest)
            rmseBest = rmseTeVector(i);
        end
    end
    
    [pairsI1, pairsJ1, pairsV1] = find(Itr);
    for k=1:trainT
        i = pairsI1(k);
        j = pairsJ1(k);
        prediction = U(:, i)' * M(:, j);
        pairsV1(k) = prediction;
    end
    predictionsTr = sparse(pairsI1, pairsJ1, pairsV1, numOfUsers, numOfItems);
    
    [pairsI2, pairsJ2, pairsV2] = find(Ite);
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        prediction = U(:, i)' * M(:, j);
        pairsV2(k) = prediction;
    end
    predictionsTe = sparse(pairsI2, pairsJ2, pairsV2, numOfUsers, numOfItems);
    
end
