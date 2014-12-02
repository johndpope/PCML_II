function [rmseStart, rmseEnd, rmseBest] = modelLambdaBias(Ytr, Yte, G, seed, numOfLatentFactors, lambda, lambdaT, lambdaB, friendshipIncludedInTheModel, alpha, numOfIterations)

    % numOfLatentFactors (default = 10)
    % alpha is the GD convergence factor (default = 0.1)
    % numOfIterations for the GD (default = 100)
    % lambda is the penalization added to the U and M (default = 1000)

    setSeed(seed);
    numOfUsers = size(Ytr, 1);
    numOfItems = size(Ytr, 2);

    [pairsI1, pairsJ1, pairsV1] = find(Ytr);
    trainT = length(pairsI1);
    
    [pairsI2, pairsJ2, pairsV2] = find(Yte);
    testT = length(pairsI2);
    
    fprintf('%d ratings in the train set.\n', trainT);
    fprintf('%d ratings in the test set.\n', testT);

    % we initialize our solution by random
    U = randn(numOfLatentFactors, numOfUsers);
    M = randn(numOfLatentFactors, numOfItems);
    
    % we initialize biases by random
    bu = randn(numOfUsers, 1);
    bm = randn(numOfItems, 1);
    
    % we make a new matrix where we put 1 if there is rating and 0 if there is
    % no ranking
    I = Ytr;
    [pairsI, pairsJ, pairsV] = find(I);
    T = length(pairsI);
    for i=1:T
        pairsV(i) = 1;
    end
    I = sparse(pairsI, pairsJ, pairsV, numOfUsers, numOfItems);

    numOfRatingsPerUser = zeros(numOfUsers, 1);
    numOfRatingsPerItem = zeros(numOfItems, 1);

    for i=1:numOfUsers
        numOfRatingsPerUser(i) = length(find(I(i, :) == 1));
    end

    for j=1:numOfItems
        numOfRatingsPerItem(j) = length(find(I(:, j) == 1));
    end

    % let's calculate RMSE
    rmse = 0.0;

    for k=1:trainT
        i = pairsI1(k);
        j = pairsJ1(k);
        r = pairsV1(k);
        prediction = bu(i) + bm(j) + U(:, i)' * M(:, j);
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmseTr1 = sqrt(rmse/trainT);

    rmse = 0;
    for i=1:numOfUsers
        Ri = Ytr(i, :);
        Ui = U(:, i);
        dg = spdiag(I(i, :) * speye(numOfItems));
        rmse = rmse + (Ri - bu(i) - bm' - Ui'*M*dg)*(Ri - bu(i) - bm' - Ui'*M*dg)';
    end
    rmseTr2 = sqrt(rmse/trainT);

    rmse = 0;
    for j=1:numOfItems
        Rj = Ytr(:, j);
        Mj = M(:, j);
        dg = spdiag(I(:, j)'*speye(numOfUsers));
        rmse = rmse + (Rj' - bu' - bm(j) - Mj'*U*dg)*(Rj' - bu' - bm(j) - Mj'*U*dg)';
    end
    rmseTr3 = sqrt(rmse/trainT);
    
    if ((abs(rmseTr1 - rmseTr2) < 0.1)&&(abs(rmseTr2 - rmseTr3) < 0.1))
        fprintf('Validity test was successfull.\n');
    else
        fprintf('Error! Validity test was NOT successfull.\n');
    end
    
    rmse = 0;
    for k=1:testT
        i = pairsI2(k);
        j = pairsJ2(k);
        r = pairsV2(k);
        prediction = bu(i) + bm(j) + U(:, i)' * M(:, j);
        rmse = rmse + (prediction - r) * (prediction - r);
    end
    rmseTe = sqrt(rmse/testT);
    
    fprintf('Before model building\n');
    fprintf('train: %f, test: %f\n', rmseTr1, rmseTe);
    
    rmseStart = rmseTe;
    
    rmseTrVector = zeros(numOfIterations, 1);
    rmseTeVector = zeros(numOfIterations, 1);

    % precalculations
    lfeye = lambda * speye(numOfLatentFactors);
    lfTeye = lambdaT * speye(numOfLatentFactors);
    
    % whether we penalize the first coefficient or not
    %lfeye(1, 1) = lambda/10;
    
    sni = speye(numOfItems);
    snu = speye(numOfUsers);
    
    fprintf('Model building phase\n');
    
    tic
    
    for iteration=1:numOfIterations
        
        fprintf('Iteration %d\n', iteration);
        newU = zeros(numOfLatentFactors, numOfUsers);
        newM = zeros(numOfLatentFactors, numOfItems);

        % for each user
        for i=1:numOfUsers
            Ri = Ytr(i, :);
            Ui = U(:, i);
            dg = spdiag(I(i, :) * sni);
            ni = sum(I(i, :));
            MMt = M * dg * dg * M';
            uRi = Ri' - bu(i) - bm;
            if (friendshipIncludedInTheModel == 1)
                g = -M * dg * uRi + (MMt + ni * lfeye) * Ui + lambdaT * (Ui - U * G(:, i));     % added for friendship graph
                H = MMt + ni * lfeye + lfTeye;
            else
                g = -M * dg * uRi + (MMt + ni * lfeye) * Ui;
                H = MMt + ni * lfeye;
            end
                        
            if (rank(H) == numOfLatentFactors)
                d = H \ g;
                newU(:, i) = U(:, i) - alpha * d;
            else
                newU(:, i) = U(:, i) - alpha * g;
            end
        end

        % for each item
        for j=1:numOfItems
            Rj = Ytr(:, j);
            Mj = M(:, j);
            dg = spdiag(I(:, j)' * snu);
            nj = sum(I(:, j));
            UUt = U * dg * dg * U';
            uRj = Rj - bm(j) - bu;
            g = -U * dg * uRj + (UUt + nj * lfeye) * Mj;
            H = UUt + nj * lfeye;
            if (rank(H) == numOfLatentFactors)
                d = H \ g;
                newM(:, j) = M(:, j) - alpha * d;
            else
                newM(:, j) = M(:, j) - alpha * g;
            end
        end
        
        g = numOfItems * bu;
    
        for j=1:numOfItems
            Rj = Ytr(:, j);
            Mj = M(:, j);
            dg = spdiag(I(:, j)'*snu);
            g = g - (Rj - bm(j) - (Mj'*U*dg)');
        end
        g = g + lambdaB * numOfItems * snu * bu; 
        newBu = bu - (alpha/1000) * g;
    
        g = numOfUsers * bm;
    
        for i=1:numOfUsers
            Ri = Ytr(i, :);
            Ui = U(:, i);
            dg = spdiag(I(i, :) * sni);
            g = g - (Ri' - bu(i) - (Ui'*M*dg)');        
        end
        g = g + lambdaB * numOfUsers * sni * bm;
        newBm = bm - (alpha/1000) * g;
        
        fprintf('Mean of bu: %f, mean of bm: %f.\n', mean(bu), mean(bm));
        
        U = newU;
        M = newM;
        bu = newBu;
        bm = newBm;
        
        % let's calculate RMSE
        rmse = 0;
        for k=1:trainT
            i = pairsI1(k);
            j = pairsJ1(k);
            r = pairsV1(k);
            prediction = bu(i) + bm(j) + U(:, i)' * M(:, j);
            rmse = rmse + (prediction - r) * (prediction - r);
        end
        rmse = sqrt(rmse/trainT);
        rmseTrVector(iteration) = rmse;

        rmse = 0;
        for k=1:testT
            i = pairsI2(k);
            j = pairsJ2(k);
            r = pairsV2(k);
            prediction = bu(i) + bm(j) + U(:, i)' * M(:, j);
            rmse = rmse + (prediction - r) * (prediction - r);
        end
        rmse = sqrt(rmse/testT);
        rmseTeVector(iteration) = rmse;

        fprintf('Train: %f, test: %f\n', rmseTrVector(iteration), rmseTeVector(iteration));
    
    end
    
    timerVal = toc;
    fprintf('Model was built in %f seconds. Average time per iteration is %f seconds.\n', timerVal, (timerVal\numOfIterations));
    
    figure;
    plot(rmseTrVector);

    figure;
    plot(rmseTeVector);

    figure;
    plot(rmseTrVector, 'g');
    hold on;
    plot(rmseTeVector, 'r');
    
    rmseEnd = rmseTeVector(numOfIterations);
    rmseBest = rmseStart;
    for i=1:numOfIterations
        if (rmseTeVector(i) < rmseBest)
            rmseBest = rmseTeVector(i);
        end
    end
    
end
