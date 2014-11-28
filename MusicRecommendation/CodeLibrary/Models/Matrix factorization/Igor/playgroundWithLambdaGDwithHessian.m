% generate some random U vectors
numOfTrueLatentFactors = 5;
numOfUsers = 10;
numOfItems = 20;

% we generate random values for trueU and trueM
trueU = 10*rand(numOfTrueLatentFactors, numOfUsers);
trueM = 10*rand(numOfTrueLatentFactors, numOfItems);

trainRatingsRatio = 0.1;
testRatingsRatio = 0.01;

trainTrueR = zeros(numOfUsers, numOfItems);
testTrueR = zeros(numOfUsers, numOfItems);

for i=1:numOfUsers
    for j=1:numOfItems
        generatedNumber = rand();
        if (generatedNumber < trainRatingsRatio)
            trainTrueR(i, j) = trueU(:, i)' * trueM(:, j);
        else
            if (generatedNumber < (trainRatingsRatio + testRatingsRatio))
                testTrueR(i, j) = trueU(:, i)' * trueM(:, j);
            end
        end
    end
end

for i=1:numOfUsers
    makeChange = 1;
    for j=1:numOfItems
        if (trainTrueR(i, j) > 0)
            makeChange = 0;
            break;
        end
    end
    if (makeChange == 1)
        selected = floor(rand()*numOfItems) + 1;
        %[selected, numOfItems]
        trainTrueR(i, selected) = trueU(:, i)' * trueM(:, selected);
    end
end

for j=1:numOfItems
    makeChange = 1;
    for i=1:numOfUsers
        if (trainTrueR(i, j) > 0)
            makeChange = 0;
            break;
        end
    end
    if (makeChange == 1)
        selected = floor(rand()*numOfUsers) + 1;
        %[selected, numOfUsers]
        trainTrueR(selected, j) = trueU(:, selected)' * trueM(:, j);
    end
end

% we define the number of iterations
numOfLatentFactors = 5;

% we initialize our solution by random
U = rand(numOfLatentFactors, numOfUsers);
M = rand(numOfLatentFactors, numOfItems);

% we make a new matrix where we put 1 if there is rating and 0 if there is
% no ranking
I = zeros(numOfUsers, numOfItems);
for i=1:numOfUsers
    for j=1:numOfItems
        if (trainTrueR(i, j) > 0)
            I(i, j) = 1;
        end
    end
end

numOfRatingsPerUser = zeros(numOfUsers, 1);
numOfRatingsPerItem = zeros(numOfUsers, 1);

for i=1:numOfUsers
    numOfRatingsPerUser(i) = sum(I(i, :));
end

for j=1:numOfItems
    numOfRatingsPerItem(j) = sum(I(:, j));
end


% let's calculate RMSE
rmse = 0;
for i=1:numOfUsers
    sumUser = 0;
    for j=1:numOfItems
        if (I(i, j) == 1)
            prediction = U(:, i)' * M(:, j);
            sumUser = sumUser + (prediction - trainTrueR(i, j)) * (prediction - trainTrueR(i, j));
            rmse = rmse + (prediction - trainTrueR(i, j)) * (prediction - trainTrueR(i, j));
        end
    end
    %fprintf('SUM USER: %f\n', sumUser);
end
    
fprintf('%f\n', rmse);

rmse = 0;
for i=1:numOfUsers
    %predictionError = trainTrueR(i, :) - (U(:, i)' * M) * diag(I(i, :) * eye(numOfItems));
    %rmse = rmse + predictionError * predictionError';
    
    Ri = trainTrueR(i, :);
    Ui = U(:, i);
    dg = diag(I(i, :) * eye(numOfItems));
    rmse = rmse + (Ri - Ui'*M*dg)*(Ri - Ui'*M*dg)';
    %rmse = rmse + Ri*Ri' - Ri*dg*M'*Ui - Ui'*M*dg*Ri'+Ui'*M*dg*dg*M'*Ui;
end

fprintf('%f\n', rmse);

rmse = 0;
for j=1:numOfItems
    
    Rj = trainTrueR(:, j);
    Mj = M(:, j);
    dg = diag(I(:, j)'*eye(numOfUsers));
    %rmse = rmse + (Rj' - Mj'*U*dg)*(Rj' - Mj'*U*dg)';
    rmse = rmse + Rj'*Rj - Rj'*dg*U'*Mj-Mj'*U*dg*Rj+Mj'*U*dg*dg*U'*Mj;
end

fprintf('%f\n', rmse);

% U
% M

alpha = 0.1;
numOfIterations = 100;
lambda = 0.001;

rmsePlot = zeros(numOfIterations, 1);

for iteration=1:numOfIterations
    fprintf('Iteration %d\n', iteration);
    
    newU = zeros(numOfLatentFactors, numOfUsers);
    
    % for each user
    for i=1:numOfUsers
        Ri = trainTrueR(i, :);
        Ui = U(:, i);
        dg = diag(I(i, :) * eye(numOfItems));
        ni = sum(I(i, :));
        g = -M * dg * Ri' + (M * dg * dg * M' + ni * lambda * eye(numOfLatentFactors)) * Ui;
        H = M * dg * dg * M' + ni * lambda * eye(numOfLatentFactors);
        d = H \ g;
        newU(:, i) = U(:, i) - alpha * d;
    end
    
    newM = zeros(numOfLatentFactors, numOfItems);
    
    for j=1:numOfItems
        Rj = trainTrueR(:, j);
        Mj = M(:, j);
        dg = diag(I(:, j)'*eye(numOfUsers));
        nj = sum(I(:, j));
        g = -U * dg * Rj + (U * dg * dg * U' + nj * lambda * eye(numOfLatentFactors)) * Mj;
        H = U * dg * dg * U' + nj * lambda * eye(numOfLatentFactors);
        d = H \ g;
        newM(:, j) = M(:, j) - alpha * d;
    end
    
    U = newU;
    M = newM;
    
    % let's calculate RMSE
    rmse = 0;
    for i=1:numOfUsers
        for j=1:numOfItems
            if (I(i, j) == 1)
                prediction = U(:, i)' * M(:, j);
                rmse = rmse + (prediction - trainTrueR(i, j)) * (prediction - trainTrueR(i, j));
            end
        end
    end
    
    fprintf('%f\n', rmse);
    
    if (rmse < 0.000001)
        fprintf('The algorithm has converged in %d iterations.\n', iteration);
        break;
    end
    
    rmsePlot(iteration) = rmse;
    
end

figure;
plot(rmsePlot);
