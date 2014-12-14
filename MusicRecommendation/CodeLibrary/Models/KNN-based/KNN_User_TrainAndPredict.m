function [ TrainPredicted, TestPredicted ] = ...
    KNN_User_TrainAndPredict(...
    Ytrain,...      % Matrix of user - artist listen counts in train    
    Ytest,...       % Indices, for which we are insterested in answer
    Gtrain,...
I_tr_all,...
I_te_all)
    % See parameters to JavaMatlabLink function for clarification
    [N, varargin] = varargGet('Alg_N', varargin);
    cv = corrcov(cov(Ytrain'));
    cv(linspace(1 , numel(cv), size(cv, 2))) = 0;
    
    TrainPredicted = Ytrain;
    TestPredicted = Ytest;
    Ypos = (Ytrain > 0);
    
    for i=1:size(Ytrain,1)
        [S,I] = sort(cv(i,:),'descend');
        I_tr = find(abs(Ytrain(i,:)) > 1e-9);
        I_te = find(abs(Ytest(i,:)) > 1e-9);
     %   TrainPredicted(i, I_tr) = (S(1:N) * Ytrain(I(1:N), I_tr)) ./ (S(1:N) * Ypos(I(1:N), I_tr));
     %   TrainPredicted(i, I_tr(find(abs((S(1:N) * Ypos(I(1:N), I_tr))) < 1e-9))) = 0;
     %   TestPredicted(i, I_te) = S(1:N) * Ytrain(I(1:N), I_te) ./ (S(1:N) * Ypos(I(1:N), I_te));
     %   TestPredicted(i, I_te(find(abs((S(1:N) * Ypos(I(1:N), I_te))) < 1e-9))) = 0;
        TrainPredicted(i, I_tr) = mean(Ytrain(I(1:N), I_tr), 1);
        TestPredicted(i, I_te) = mean(Ytrain(I(1:N), I_te), 1);
    end    
    TrainPredicted(isnan(TrainPredicted)) = 0;
    TestPredicted(isnan(TestPredicted)) = 0;
    TrainPredicted(isinf(TrainPredicted)) = 0;
    TestPredicted(isinf(TestPredicted)) = 0;
end

