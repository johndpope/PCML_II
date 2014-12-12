function rmse = calculateRMSE(predicted_values, true_values, I)
    [pairsI, pairsJ, pairsV] = find(I);
    T = length(pairsI);
    rmse = 0;
    for k=1:T
        i = pairsI(k);
        j = pairsJ(k);
        true_value = true_values(i, j);
        predicted_value = predicted_values(i, j);
        rmse = rmse + (predicted_value - true_value) * (predicted_value - true_value);
    end
    rmse = sqrt(rmse / T);
end
