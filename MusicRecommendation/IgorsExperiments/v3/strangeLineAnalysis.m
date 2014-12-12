load('ALS.mat');    % exp_1

YteTrue1 = YteTransformed;
YtePredicted1 = Pte0;
Ite1 = Ite;
U1 = U;
M1 = M;
YtePredictedFinal1 = Pte2;

load('ALS_exponential_data.mat');   % exp_2

YteTrue2 = YteTransformed;
YtePredicted2 = Pte0;
Ite2 = Ite;
U2 = U;
M2 = M;
YtePredictedFinal2 = Pte2;

% the same identity matrix
Ite = Ite1;

% the same true values
YteTrue = YteTrue2;

% let's put all values in arrays
arrPredicted1 = allRatingsInArray(YtePredicted1, Ite);
arrPredicted2 = allRatingsInArray(YtePredicted2, Ite);
arrTrue = allRatingsInArray(YteTrue, Ite);
arrPredictedFinal1 = allRatingsInArray(YtePredictedFinal1, Ite);
arrPredictedFinal2 = allRatingsInArray(YtePredictedFinal2, Ite);
T = length(arrTrue);

% lets's find the most frequent value in both arrays
[value1, longest1] = findTheMostFrequentValue(arrPredicted1);
fprintf('The most frequent value is %f. It appears %d times.\n', value1, longest1);
[value2, longest2] = findTheMostFrequentValue(arrPredicted2);
fprintf('The most frequent value is %f. It appears %d times.\n', value2, longest2);

numOfUsers = size(Ite, 1);
numOfItems = size(Ite, 2);

% find famous artists
famous = findFamous(Ytr, Itr, 500);
labels = zeros(T, 1);
[pi, pj, pr] = find(Ite);
for k=1:T
    labels(k) = famous(pj(k));
end

labelsUsers = zeros(numOfUsers, 1);
labelsItems = zeros(numOfItems, 2);
labelsArr = zeros(T, 1);

figure;
scatter(arrPredicted1(find(labels == 1)), arrPredicted2(find(labels == 1)));

figure;
scatter(arrPredicted2(find(labels == 1)), arrTrue(find(labels == 1)));

figure;
scatter(arrPredicted1(find(labels == 1)), arrTrue(find(labels == 1)), [], [1, 0, 0]);

np = zeros(T, 1);
for k=1:T
    np(k) = 0.5 * arrPredictedFinal1(k) + 0.5 * arrPredictedFinal2(k);
end

figure;
scatter(np, allRatingsInArray(Yte, Ite), [], [1, 0, 0]);
xlim([1, 1000]);
ylim([1, 1000]);

rmse = calculateRMSE(YtePredictedFinal1, Yte, Ite);
fprintf('RMSE1: %f.\n', rmse);
rmse = calculateRMSE(YtePredictedFinal2, Yte, Ite);
fprintf('RMSE2: %f.\n', rmse);

NP = generateMatrixFromArray(np, Ite);
rmse = calculateRMSE(NP, Yte, Ite);
fprintf('Final RMSE: %f.\n', rmse);
[pi, pj, pr] = find(Ite);
% special case 
for k=1:T
    if ((arrPredicted1(k) == 1)&&(arrTrue(k) >= 8))
        fprintf('YES %d %d.\n', pi(k), pj(k));
    end
end

rmse10 = 0;
rmse11 = 0;
rmse20 = 0;
rmse21 = 0;

count0 = 0;
count1 = 0;

for k=1:T
    i = pi(k);
    j = pj(k);
    if (labels(k) == 0)
        rmse10 = rmse10 + (YtePredictedFinal1(i, j) - Yte(i, j)) * (YtePredictedFinal1(i, j) - Yte(i, j));
        rmse20 = rmse20 + (YtePredictedFinal2(i, j) - Yte(i, j)) * (YtePredictedFinal2(i, j) - Yte(i, j));
        count0 = count0 + 1;
    else
        rmse11 = rmse11 + (YtePredictedFinal1(i, j) - Yte(i, j)) * (YtePredictedFinal1(i, j) - Yte(i, j));
        rmse21 = rmse21 + (YtePredictedFinal2(i, j) - Yte(i, j)) * (YtePredictedFinal2(i, j) - Yte(i, j));        
        count1 = count1 + 1;
    end
end
rmse10 = sqrt(rmse10 / count0);
rmse20 = sqrt(rmse20 / count0);
rmse11 = sqrt(rmse11 / count1);
rmse21 = sqrt(rmse21 / count1);
fprintf('%f %f %f %f.\n', rmse10, rmse20, rmse11, rmse21);
