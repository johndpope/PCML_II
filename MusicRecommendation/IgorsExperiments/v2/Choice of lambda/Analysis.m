load('matlab.mat');

iterationThreshold = 50;

rmseTrTr = zeros(NL, 1);
rmseTeTe = zeros(NL, 1);
rmseBaTr = zeros(NL, 1);
rmseBaTe = zeros(NL, 1);

for i=1:NL
    
    for j=1:numOfTrials
        
        rmseTrTr(i) = rmseTrTr(i) + rmseTrVector2(i, j, iterationThreshold);
        rmseTeTe(i) = rmseTeTe(i) + rmseTeVector2(i, j, iterationThreshold);
        
        rmseBaTr(i) = rmseBaTr(i) + rmseTrBaselineVector(i, j);
        rmseBaTe(i) = rmseBaTe(i) + rmseTeBaselineVector(i, j);
        
    end
    
    rmseTrTr(i) = rmseTrTr(i) / numOfTrials;
    rmseTeTe(i) = rmseTeTe(i) / numOfTrials;
    
    rmseBaTr(i) = rmseBaTr(i) / numOfTrials;
    rmseBaTe(i) = rmseBaTe(i) / numOfTrials;
    
    fprintf('lambda: %f, baseline: %f, our model: %f.\n', lambdas(i), rmseBaTe(i), rmseTeTe(i));
    
end

% figure;
% plot(rmseBaTe(3:end, 1));
% figure;
% plot(rmseTeTe(3:end, 1));

for i=1:NL
    lambdaID = i;
    figure;
    %str = sprintf('lambda: %f, average RMSE baseline: %f, average RMSE our model: %f.\n', lambdas(lambdaID), rmseBaTe(i), rmseTeTe(i));
    for j=1:numOfTrials
        trialID = j;
        
        subplot(4, 3, j)
        dummyArr = zeros(length(rmseTeVector2(lambdaID, trialID, 3:end)), 1);
        for k=1:size(dummyArr, 1)
            dummyArr(k) = rmseTeBaselineVector(lambdaID, trialID);
        end
        plot(dummyArr);
        hold on;
        plot(reshape(rmseTeVector2(lambdaID, trialID, 3:end), length(rmseTeVector2(lambdaID, trialID, 3:end)), 1), '--g');
    end

end
