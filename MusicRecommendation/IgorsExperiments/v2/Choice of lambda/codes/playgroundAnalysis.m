% we are working with Ytr, Yte, Itr, Ite
% pYte0 (baseline predictions) and Pte2 (predictions of our model)
load('data.mat');

rmseMyModel = calculateRMSE(Pte2, Yte, Ite);
rmseBaseline = calculateRMSE(pYte0, Yte, Ite);

valuesTrain = allRatingsInArray(Ytr, Itr);
valuesMyModel = allRatingsInArray(Pte2, Ite);
valuesBaseline = allRatingsInArray(pYte0, Ite);
valuesTrue = allRatingsInArray(Yte, Ite);

totalMyModel = length(valuesMyModel);
totalBaseline = length(valuesBaseline);

fprintf('==============================\n');
fprintf('RMSE of my model: %f.\n', rmseMyModel);
fprintf('RMSE of baseline model: %f.\n', rmseBaseline);
fprintf('==============================\n');

% let's calculate the rmse for different bins
binsBorders = [100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 100000, 1000000];
numberOfBins = length(binsBorders);

rmsesMyModel = zeros(numberOfBins, 1);
rmsesBaseline = zeros(numberOfBins, 1);

totalsMyModel = zeros(numberOfBins, 1);
totalsBaseline = zeros(numberOfBins, 1);

fprintf('\n==============================\n');
last = 0;
for k=1:numberOfBins
    [rmsesMyModel(k), totalsMyModel(k)] = calculateRMSEforRange(Pte2, Yte, Ite, last, binsBorders(k));
    [rmsesBaseline(k), totalsBaseline(k)] = calculateRMSEforRange(pYte0, Yte, Ite, last, binsBorders(k));
    
    contributesMyModel = ((rmsesMyModel(k) * rmsesMyModel(k)) * totalsMyModel(k)) / (rmseMyModel * rmseMyModel * totalMyModel);
    shouldContributeMyModel = totalsMyModel(k) / totalMyModel;
    percentageMyModel = (contributesMyModel / shouldContributeMyModel) * 100;
    
    contributesBaseline = ((rmsesBaseline(k) * rmsesBaseline(k)) * totalsBaseline(k)) / (rmseBaseline * rmseBaseline * totalBaseline);
    shouldContributeBaseline = totalsBaseline(k) / totalBaseline;
    percentageBaseline = (contributesBaseline / shouldContributeBaseline) * 100;
    
    fprintf('Range (%d, %d]\n', last, binsBorders(k));
    fprintf('MyModel. Total: %d. RMSE: %.3f. Contribution: %.3f.\n', totalsMyModel(k), rmsesMyModel(k), percentageMyModel);
    fprintf('Baseline. Total: %d. RMSE: %.3f. Contribution: %.3f.\n', totalsBaseline(k), rmsesBaseline(k), percentageBaseline);
    last = binsBorders(k);
end
fprintf('==============================\n');

rmsesMyModel = zeros(numberOfBins, 1);
rmsesBaseline = zeros(numberOfBins, 1);

totalsMyModel = zeros(numberOfBins, 1);
totalsBaseline = zeros(numberOfBins, 1);

fprintf('\n==============================\n');
last = 0;
for k=1:numberOfBins
    [rmsesMyModel(k), totalsMyModel(k)] = calculateRMSEforRange(Yte, Pte2, Ite, last, binsBorders(k));
    [rmsesBaseline(k), totalsBaseline(k)] = calculateRMSEforRange(Yte, pYte0, Ite, last, binsBorders(k));
    
    contributesMyModel = ((rmsesMyModel(k) * rmsesMyModel(k)) * totalsMyModel(k)) / (rmseMyModel * rmseMyModel * totalMyModel);
    shouldContributeMyModel = totalsMyModel(k) / totalMyModel;
    percentageMyModel = (contributesMyModel / shouldContributeMyModel) * 100;
    
    contributesBaseline = ((rmsesBaseline(k) * rmsesBaseline(k)) * totalsBaseline(k)) / (rmseBaseline * rmseBaseline * totalBaseline);
    shouldContributeBaseline = totalsBaseline(k) / totalBaseline;
    percentageBaseline = (contributesBaseline / shouldContributeBaseline) * 100;
    
    fprintf('Range (%d, %d]\n', last, binsBorders(k));
    fprintf('MyModel. Total: %d. RMSE: %.3f. Contribution: %.3f.\n', totalsMyModel(k), rmsesMyModel(k), percentageMyModel);
    fprintf('Baseline. Total: %d. RMSE: %.3f. Contribution: %.3f.\n', totalsBaseline(k), rmsesBaseline(k), percentageBaseline);
    last = binsBorders(k);
end
fprintf('==============================\n');


valuesMyModel = allRatingsInArray(Pte2, Ite);
valuesBaseline = allRatingsInArray(pYte0, Ite);
valuesTrue = allRatingsInArray(Yte, Ite);

xMaxRange = 5000;
yMaxRange = 5000;

figure;
scatter(valuesTrue, valuesMyModel);
xlim([0, xMaxRange]);
ylim([0, yMaxRange]);
title('True vs. MyModel');

figure;
scatter(valuesTrue, valuesBaseline);
xlim([0, xMaxRange]);
ylim([0, yMaxRange]);
title('True vs. Baseline');

figure;
scatter(valuesMyModel, valuesTrue);
xlim([0, xMaxRange]);
ylim([0, yMaxRange]);
title('MyModel vs. True');

figure;
scatter(valuesBaseline, valuesTrue);
xlim([0, xMaxRange]);
ylim([0, yMaxRange]);
title('Baseline vs. True');

% there is a strange case where one value repeats a lot in the predictions

[value, count] = findTheMostFrequentValue(valuesMyModel);
fprintf('\n==============================\n');
fprintf('The most frequent value in MyModel is %f. It appears %d times.\n', value, count);

T = length(valuesMyModel);
count = 0;
for k=1:T
    if (abs(valuesMyModel(k) - value) < 30)
        count = count + 1;
        selectedTrueValues(count) = valuesTrue(k);
        selectedBaselineValues(count) = valuesBaseline(k);
        updatedValuesMyModel(k) = valuesBaseline(k);
    else
        updatedValuesMyModel(k) = valuesMyModel(k);
    end
end

figure;
hist(selectedTrueValues, 100);
title('Selected true values');

uPte2 = generateMatrixFromArray(updatedValuesMyModel, Ite);
rmse = calculateRMSE(uPte2, Yte, Ite);
fprintf('RMSE of the updated model is: %f.\n', rmse);

figure;
scatter(updatedValuesMyModel, valuesTrue);
xlim([0, xMaxRange]);
ylim([0, yMaxRange]);
title('MyModelUpdated vs. True');

fprintf('==============================\n');

numOfUsers = size(Yte, 1);
numOfItems = size(Yte, 2);

averageNumberOfCounts = zeros(numOfItems, 1);
for j=1:numOfItems
    if (sum(sum(Itr(:, j))) == 0)
        averageNumberOfCounts(j) = -1;
    else
        averageNumberOfCounts(j) = sum(sum(Ytr(:, j))) / sum(sum(Itr(:, j)));
    end
end

famousArtists = zeros(numOfItems, 1);
threshold = 1000;
for j=1:numOfItems
    if (averageNumberOfCounts(j) > threshold)
        famousArtists(j) = 1;
    end
end

[pi, pj, pr] = find(Itr);
T = length(valuesTrue);
labels = zeros(T, 1);
for k=1:T
    if (famousArtists(pj(k)) == 1)
        labels(k) = 1;
    end
end

figure;
scatter(valuesMyModel(find(labels == 0)), valuesTrue(find(labels == 0)));
xlim([0, 1000]);
ylim([0, 1000]);

figure;
scatter(valuesMyModel(find(labels == 1)), valuesTrue(find(labels == 1)), [], [1, 0, 0]);
xlim([0, 1000]);
ylim([0, 1000]);

[pi, pj, pr] = find(Ite);

% if we remove the problematic values what is the baseline and what are our
% predictions
% pYte0 (baseline predictions) and Pte2 (predictions of our model)
T = length(valuesMyModel);
count = 0;
for k=1:T
    if (abs(valuesMyModel(k) - value) < 25)
        pYte0(pi(k), pj(k)) = 0;
        Pte0(pi(k), pj(k)) = 0;
        Ite(pi(k), pj(k)) = 0;
        Yte(pi(k), pj(k)) = 0;
    end
end

rmseMyModel = calculateRMSE(Pte2, Yte, Ite);
rmseBaseline = calculateRMSE(pYte0, Yte, Ite);
fprintf('MyModel: %f, Baseline: %f.\n', rmseMyModel, rmseBaseline);
