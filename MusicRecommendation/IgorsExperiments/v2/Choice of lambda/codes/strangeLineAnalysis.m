load('exp_1.mat');

YteTrue1 = YteTransformed;
YtePredicted1 = Pte0;
Ite1 = Ite;
U1 = U;
M1 = M;
YtePredictedFinal1 = Pte2;

load('exp_2.mat');

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

% the same baseline


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

labelsUsers = zeros(numOfUsers, 1);
labelsItems = zeros(numOfItems, 2);
labelsArr = zeros(T, 1);

[pi, pj, pr] = find(Ite);

count = 0;
for k=1:T
    if ((arrPredicted1(k) == value1)&&(arrPredicted2(k) == value2))
        % the second condition is always true is the first one is true
        if ((isZeroVector(U1(:, pi(k))) == 1)&&(isZeroVector(M1(:, pj(k))) == 1)&&(isZeroVector(U2(:, pi(k))) == 1)&&(isZeroVector(M2(:, pj(k))) == 1))
            labelsUsers(pi(k)) = 1;
            labelsItems(pj(k)) = 1;
            labelsArr(k) = 1;
        end
    end
end

figure;
scatter(arrPredicted1, arrPredicted2);

% the second predictions are on logaritmized data
arrFinalPredictions = arrPredicted2;
for k=1:T
    if (labelsArr(k) == 1)
        
        ss1 = length(find(Itr(pi(k), :) == 1));
        ss2 = length(find(Itr(:, pj(k)) == 1));
        
        fprintf('%f %d %d.\n', arrPredictedFinal2(k), ss1, ss2);
        arrFinalPredictions(k) = log(arrPredictedFinal2(k));
    end
end

figure;
scatter(arrFinalPredictions, arrTrue);
% xlim([0, 1000]);
% ylim([0, 1000]);
fprintf('test: %f.\n', calculateRMSE(Pte2, Yte, Ite));