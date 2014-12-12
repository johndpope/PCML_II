load('songTrain.mat');

% we copy the data
Y = Ytrain;
G = Gtrain;

% we remove the empty rows and columns
[Y, G] = removeEmptyRowsAndColumns(Y, getI(Y), G);

% number of trials
numOfTrials = 1;

rmseTrainBaseline = 0;
rmseTestBaseline = 0;
rmseTrainMyModel = 0;
rmseTestMyModel = 0;

seeds = [1, 3, 6, 21, 42, 56, 89, 122, 8392, 192921];
seeds = seeds + 78;

for trial=1:numOfTrials
    seed = seeds(trial);
    
    fprintf('\nTrial %d\n', trial);
    
    % we split the data into two sets
    % [Ytr, Yte] = getSplitDataWeak(Y, G, seed, 10);
    [Ytr, Yte] = getSplitDataWeak2(Y, seed, 0.9);

    checkValidity(Ytr);

    % we get the indicator matrices for Ytr and Yte
    Itr = getI(Ytr);
    Ite = getI(Yte);

    fprintf('==============================\n');
    fprintf('Number of ratings in the train dataset: %d.\n', length(find(Itr > 0)));
    fprintf('Number of ratings in the test dataset: %d.\n', length(find(Ite > 0)));
    fprintf('==============================\n');

    % we obtain the baseline predictions
    [pYtr0, pYte0, rmse1, rmse2, rmse3] = getBaseLinePredictions(Ytr, Yte, 0.5);
    
    fprintf('\n==============================\n');
    fprintf('Baseline prediction - user average: %f.\n', rmse1);
    fprintf('Baseline prediction - artist average: %f.\n', rmse2);
    fprintf('Baseline prediction - weighted user and artist average: %f.\n', rmse3);
    fprintf('==============================\n');
    
    rmseTrainBaseline = rmseTrainBaseline + calculateRMSE(pYtr0, Ytr, Itr);
    rmseTestBaseline = rmseTestBaseline + calculateRMSE(pYte0, Yte, Ite);
    
    fprintf('\n==============================\n');
    fprintf('Baseline prediction - weighted user and artist average\n');
    fprintf('Train: %f, test: %f.\n', calculateRMSE(pYtr0, Ytr, Itr), calculateRMSE(pYte0, Yte, Ite));
    fprintf('Logarithmized baseline. Train: %f, test: %f.\n', calculateRMSE(logaritmizeSparseMatrix(pYtr0), logaritmizeSparseMatrix(Ytr), Itr), calculateRMSE(logaritmizeSparseMatrix(pYte0), logaritmizeSparseMatrix(Yte), Ite));
    fprintf('==============================\n');

    % we transform the ratings (the ratings should not be changed)

    % option 1
    % [YtrTransformed, lambda] = transformWithBoxCox(Ytr);
    % YteTransformed = transformWithBoxCoxLambda(lambda, Yte);

    % option 2
    % [YtrTransformed, lambda] = transformWithBoxCoxRows(Ytr, getI(Ytr));
    % YteTransformed = transformWithBoxCoxLambdaRows(Yte, getI(Yte), lambda);

    % option 3
    YtrTransformed = logaritmizeSparseMatrix(Ytr);
    YteTransformed = logaritmizeSparseMatrix(Yte);

    % we normalize the ratings
    
    % option 1
    % [YtrTransformed, YteTransformed, meanY, stdY] = normalizeMatrix(YtrTransformed, YteTransformed, Itr, Ite);

    % option 2
    % [YtrTransformed, YteTransformed, meanY] = normalizeMatrixMean(YtrTransformed, YteTransformed, Itr, Ite);

    % option 3
    % we do nothing

    % =============== OUR MODEL HERE ===============
    % Input: YtrTransformed, YteTransformed, Itr, Ite, G
    % Output: Ptr0, Pte0
    fprintf('\n==============================\n');
    fprintf('Output from our model\n');

    % parameters for our model
    numOfLatentFactors = 10;
    lambdaU = 0.08;             % 180 for method 1
    lambdaM = 0.08;             % 180 for method 1
    %lambdaU = 0.08;             % 0.03 for method 1
    %lambdaM = 0.08;             % 0.03 for method 1
    lambdaT = 0.08;
    friendshipIncluded = 1;
    numOfIterations = 30;

    fprintf('Number of latent factors: %d.\n', numOfLatentFactors);
    fprintf('Number of iterations: %d.\n', numOfIterations);
    if (friendshipIncluded == 1)
        fprintf('LambdaU: %f, lambdaM: %f, lambdaT: %f.\n', lambdaU, lambdaM, lambdaT);
    else
        fprintf('LambdaU: %f, lambdaM: %f.\n', lambdaU, lambdaM);
    end

    % we run our model
    
    % method 1
    [Ptr0, Pte0, rmseStart, rmseEnd, rmseBest, U, M] = modelLambdaLS(YtrTransformed, YteTransformed, Itr, Ite, G, seed, numOfLatentFactors, lambdaU, lambdaM, lambdaT, friendshipIncluded, numOfIterations);
    
    % method 2
    % weight = adjustTheWeights(Ytr, Itr, -meanY);
    %weightingRate = 1;  % from 0 to 1. 0 means we are using plain ALS 1 means we are using weighted ALS with high weights
    %weight = getWeightMatrix(Ytr, Itr, weightingRate);
    %[Ptr0, Pte0, rmseStart, rmseEnd, rmseBest, U, M] = modelWeightedLambdaLS(YtrTransformed, YteTransformed, Itr, Ite, weight, G, seed, numOfLatentFactors, lambdaU, lambdaM, lambdaT, friendshipIncluded, numOfIterations);
    
    % method 3
    % alg = {'mm', 'cjlin', 'als', 'alsobs', 'prob'};
    % YtrTransformed2 = setNegativeToZero(YtrTransformed);
    % [U, M] = nmf(YtrTransformed2, numOfLatentFactors, alg{3}, numOfIterations, 0);
    % Ptr0 = getPredictions(YtrTransformed, Itr, U, M);
    % Pte0 = getPredictions(YteTransformed, Ite, U, M);
    
    % method 4
    % Ptr0 = YtrTransformed;
    % Pte0 = YteTransformed;
    
    fprintf('==============================\n');
    % ==============================================

    
    
    figure;
    hist(allRatingsInArray(YteTransformed, Ite), 100);
    title('Distribution of the true values from the test set');

    figure;
    hist(allRatingsInArray(Pte0, Ite), 100);
    title('Distribution of the predictions for the test set');

    figure;
    scatter(allRatingsInArray(Pte0, Ite), allRatingsInArray(YteTransformed, Ite));
    xlim([-10, 10]);
    ylim([-10, 10]);
    title('Test: Predicted vs. true');

    figure;
    scatter(allRatingsInArray(Ptr0, Itr), allRatingsInArray(YtrTransformed, Itr));
    %xlim([-10, 10]);
    %ylim([-10, 10]);
    title('Train: Predicted vs. true');
    
    figure;
    scatter(allRatingsInArray(YteTransformed, Ite), allRatingsInArray(Pte0, Ite));
    xlim([-10, 10]);
    ylim([-10, 10]);
    title('Test: True vs. predicted');
    
    figure;
    avgError = zeros(size(Yte, 1), 1);
    countTot = zeros(size(Yte, 1), 1);
    [pi, pj, pr] = find(Ite);
    T = length(pi);
    for k=1:T
        countTot(pi(k)) = countTot(pi(k)) + 1;
        avgError(pi(k)) = avgError(pi(k)) + abs(Pte0(pi(k), pj(k)) - YteTransformed(pi(k), pj(k)));
    end
    for i=1:size(Yte, 1)
        avgError(i) = avgError(i) / countTot(i);
    end
    
    avgErrorSorted = sort(avgError);
    figure;
    plot(avgErrorSorted);
    title('Average error for different users');
    
    avgCount = zeros(size(Yte, 1), 1);
    countTot = zeros(size(Yte, 1), 1);
    [pi, pj, pr] = find(Itr);
    T = length(pi);
    for k=1:T
        countTot(pi(k)) = countTot(pi(k)) + 1;
        avgCount(pi(k)) = avgCount(pi(k)) + YtrTransformed(pi(k), pj(k));
    end
    for i=1:size(Yte, 1)
        avgCount(i) = avgCount(i) / countTot(i);
    end
    avgCountSorted = sort(avgCount);
    %selected = avgCountSorted(floor(length(avgCountSorted)/2));
    selected = avgCountSorted(length(avgCountSorted) - 100);
    
    figure;
    selectedAvgErrors = avgError(avgCount <= selected);
    selectedAvgErrorsSorted = sort(selectedAvgErrors);
    plot(selectedAvgErrorsSorted);
    ylim([0, 9]);
    title('Average error for one group of users');
    
    figure;
    selectedAvgErrors = avgError(avgCount > selected);
    selectedAvgErrorsSorted = sort(selectedAvgErrors);
    plot(selectedAvgErrorsSorted);
    ylim([0, 9]);
    title('Average error for other group of users');
    
    % we denormalize the ratings

    % option 1
    % [Ptr1, Pte1] = denormalizeMatrix(Ptr0, Pte0, Itr, Ite, meanY, stdY);

    % option 2
    % [Ptr1, Pte1] = denormalizeMatrixMean(Ptr0, Pte0, Itr, Ite, meanY);

    % option 3
    Ptr1 = Ptr0;
    Pte1 = Pte0;
    
    % we detransform the ratings

    % option 1
    % Ptr2 = detransformWithBoxCox(Ptr1, Itr, lambda);
    % Pte2 = detransformWithBoxCox(Pte1, Ite, lambda);

    % option 2
    % Ptr2 = detransformWithBoxCoxRows(Ptr1, Itr, lambda);
    % Pte2 = detransformWithBoxCoxRows(Pte1, Ite, lambda);

    % option 3
    Ptr2 = expSparseMatrix(Ptr1, Itr);
    Pte2 = expSparseMatrix(Pte1, Ite);

    fprintf('\n==============================\n');
    fprintf('Prediction with our model\n');
    fprintf('Train: %f, test: %f.\n', calculateRMSE(Ptr2, Ytr, Itr), calculateRMSE(Pte2, Yte, Ite));
    fprintf('Test (weighted baseline + our model): %f\n', calculateRMSE(0.5 * Pte2 + 0.5 * pYte0, Yte, Ite));
    fprintf('==============================\n');
    
    rmseTrainMyModel = rmseTrainMyModel + calculateRMSE(Ptr2, Ytr, Itr);
    rmseTestMyModel = rmseTestMyModel + calculateRMSE(Pte2, Yte, Ite);
end

rmseTrainBaseline = rmseTrainBaseline / numOfTrials;
rmseTestBaseline = rmseTestBaseline / numOfTrials;

rmseTrainMyModel = rmseTrainMyModel / numOfTrials;
rmseTestMyModel = rmseTestMyModel / numOfTrials;

fprintf('\n==============================\n');
fprintf('Average prediction with baseline\n');
fprintf('Train: %f, test: %f.\n', rmseTrainBaseline, rmseTestBaseline);
fprintf('==============================\n');

fprintf('\n==============================\n');
fprintf('Average prediction with our model\n');
fprintf('Train: %f, test: %f.\n', rmseTrainMyModel, rmseTestMyModel);
fprintf('==============================\n');
