% generate some random U vectors
numOfTrueLatentFactors = 5;
numOfUsers = 1000;
numOfItems = 15000;

% we generate random values for trueU and trueM
trueU = 10*rand(numOfTrueLatentFactors, numOfUsers);
trueM = 10*rand(numOfTrueLatentFactors, numOfItems);

trainRatingsRatio = 0.01;
testRatingsRatio = 0.001;

trainTrueR = zeros(numOfUsers, numOfItems);
testTrueR = zeros(numOfUsers, numOfItems);

% we generate sparse matrix trainTrueR and testTrueR
sampleR = sprand(numOfUsers, numOfItems, trainRatingsRatio + testRatingsRatio);
[pairsI, pairsJ, pairsV] = find(sampleR);
T = length(pairsI);
p = randperm(T);
trainT = floor((T*trainRatingsRatio)/(trainRatingsRatio+testRatingsRatio));
testT = T - trainT;

pairsI1 = zeros(trainT, 1);
pairsJ1 = zeros(trainT, 1);
pairsV1 = zeros(trainT, 1);

pairsI2 = zeros(testT, 1);
pairsJ2 = zeros(testT, 1);
pairsV2 = zeros(testT, 1);

threshold = trainT;
trainT = 0;
testT = 0;

for i=1:T
    pairsV(i) = trueU(:, pairsI(i))' * trueM(:, pairsJ(i));
    if (p(i) <= threshold)
        trainT = trainT + 1;
        pairsI1(trainT) = pairsI(i);    
        pairsJ1(trainT) = pairsJ(i);
        pairsV1(trainT) = pairsV(i);
    else
        testT = testT + 1;
        pairsI2(testT) = pairsI(i);    
        pairsJ2(testT) = pairsJ(i);
        pairsV2(testT) = pairsV(i);
    end
end

trainTrueR = sparse(pairsI1, pairsJ1, pairsV1, numOfUsers, numOfItems, trainT + numOfUsers + numOfItems);
testTrueR = sparse(pairsI2, pairsJ2, pairsV2, numOfUsers, numOfItems, testT + numOfUsers + numOfItems);

for i=1:numOfUsers
    if (isempty(find(trainTrueR(i, :) > 0)))
        selected = floor(rand()*numOfItems) + 1;
        trainTrueR(i, selected) = trueU(:, i)' * trueM(:, selected);
    end
end

for j=1:numOfItems
    if (isempty(find(trainTrueR(:, j) > 0)))
        selected = floor(rand()*numOfUsers) + 1;
        %[selected, numOfUsers]
        trainTrueR(selected, j) = trueU(:, selected)' * trueM(:, j);
    end
end

[pairsI1, pairsJ1, pairsV1] = find(trainTrueR);
trainT = length(pairsI1);
fprintf('%d ratings in the train set\n', trainT);

% our solution here
numOfLatentFactors = 5;

% we initialize our solution by random
U = rand(numOfLatentFactors, numOfUsers);
M = rand(numOfLatentFactors, numOfItems);

% we make a new matrix where we put 1 if there is rating and 0 if there is
% no ranking
I = trainTrueR;
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
rmse = 0;

trainT

for k=1:trainT
    i = pairsI1(k);
    j = pairsJ1(k);
    r = pairsV1(k);
    prediction = U(:, i)' * M(:, j);
    rmse = rmse + (prediction - r) * (prediction - r);
end
fprintf('%f\n', rmse);

rmse = 0;
for i=1:numOfUsers
    % size: 1x(numOfItems) 
    Ri = trainTrueR(i, :);
    % size: (numOfLatentFactors)x1
    Ui = U(:, i);
    dg = spdiag(I(i, :) * speye(numOfItems));
    rmse = rmse + (Ri - Ui'*M*dg)*(Ri - Ui'*M*dg)';
end
fprintf('%f\n', rmse);

rmse = 0;
for j=1:numOfItems
    Rj = trainTrueR(:, j);
    Mj = M(:, j);
    dg = spdiag(I(:, j)'*speye(numOfUsers));
    rmse = rmse + (Rj' - Mj'*U*dg)*(Rj' - Mj'*U*dg)';
end
fprintf('%f\n', rmse);

alpha = 0.0001;
numOfIterations = 10;
lambda = 0.001;

rmsePlot = zeros(numOfIterations, 1);

for iteration=1:numOfIterations
    fprintf('Iteration %d\n', iteration);
    
    newU = zeros(numOfLatentFactors, numOfUsers);
    
    % for each user
    for i=1:numOfUsers
        Ri = trainTrueR(i, :);
        Ui = U(:, i);
        dg = spdiag(I(i, :) * speye(numOfItems));
        ni = sum(I(i, :));
        g = -M * dg * Ri' + (M * dg * dg * M' + ni * lambda * speye(numOfLatentFactors)) * Ui;
        newU(:, i) = U(:, i) - alpha * g;
    end
    
    newM = zeros(numOfLatentFactors, numOfItems);
    
    for j=1:numOfItems
        Rj = trainTrueR(:, j);
        Mj = M(:, j);
        dg = spdiag(I(:, j)'*speye(numOfUsers));
        nj = sum(I(:, j));
        g = -U * dg * Rj + (U * dg * dg * U' + nj * lambda * speye(numOfLatentFactors)) * Mj;
        newM(:, j) = M(:, j) - alpha * g;
    end
    
    U = newU;
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
    fprintf('%f\n', rmse);
    
    if (rmse < 0.000001)
        fprintf('The algorithm has converged in %d iterations.\n', iteration);
        break;
    end
    
    rmsePlot(iteration) = rmse;
    
end

figure;
plot(rmsePlot);
