% Computes RMSE for only known user - artist pairs
% TODO: ask about exact procedure
function [ ret ] = RMSE(Predicted, Truth)
    Predicted = Predicted(:); 
    Truth = Truth(:);
    idx = find(Truth > 0);
    Truth = Truth(idx);
    Predicted = Predicted(idx);
    ret = sqrt((Truth - Predicted)' * (Truth - Predicted) / size(Truth, 1));
end

