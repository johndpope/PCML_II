function [ TrainPredicted, TestPredicted ] = ...
    KNN_Friend_TrainAndPredict(...
    Ytrain,...      % Matrix of user - artist listen counts in train    
    Ytest,...       % Indices, for which we are insterested in answer
    Gtrain,...             %
    I_tr_all,...
    I_te_all)

   Y_tr_0 = Ytrain;
   Y_te_0 = Ytest;

  [meanAll, meanU, meanI, Ytest] = subtractBaseline(Y_te_0, I_te_all, Y_tr_0, I_tr_all);
  [meanAll, meanU, meanI, Ytrain] = subtractBaseline(Y_tr_0, I_tr_all, Y_tr_0, I_tr_all);
  
 

    % See parameters to JavaMatlabLink function for clarification
     TrainPredicted = Ytrain;
     TestPredicted = Ytest;
    
%     cv = corrcov(cov(Ytrain'));
%    cv(linspace(1 , numel(cv), size(cv, 2))) = 0;
     Ypos = I_tr_all;

%   GG = Gtrain * Gtrain;
    ranks = pagerank(Gtrain);
%   ranks = full(sum(Gtrain, 2));
%    mul = 1;
%    cv = Gtrain * repmat(ranks, 1, size(Gtrain, 1));
    
    cv = Gtrain;
% 
%       cv = graphallshortestpaths(Gtrain);
%       cv(isinf(cv)) = 0;
%       cv(cv > 2) = 0;
%       cv = 1./ cv;
%       cv(isinf(cv)) = 0;
     
%    cv = 
%    cv = G;
%      GG = Gtrain;
%      for i = 1:1
%          GG = GG * Gtrain;
%          GG = GG / full(max(max(GG)));
%          mul = mul * 0.0;
%          cv = cv + GG * mul;
%     end
   
   for i=1:size(Ytrain,1)
        Fr = find(Gtrain(i,:) > 0);
   %     Fr2 = find(GG(:,i) > 0);
        I_tr = find(I_tr_all(i,:) > 0);
        I_te = find(I_te_all(i,:) > 0);
        
         [S,I] = sort(cv(i,:),'descend');
         
         N = size(Ytrain, 1);
         
%         tr_1 = mean(Ytrain(Fr, I_tr));
%         tr_1 = sum(Ytrain(Fr, I_tr)) ./ sum(Ytrain(Fr, I_tr) > 0);
 %        tr_2 = mean(Ytrain(Fr2, I_tr));
         tr_1 = (S(1:N) * Ytrain(I(1:N), I_tr)) / sum(S(1:N)); %./ (S(1:N) * Ypos(I(1:N), I_tr));
 %        tr_2(abs((S(1:N) * Ypos(I(1:N), I_tr))) < 1e-9) = 0;
 %        
 %        te_1 = mean(Ytrain(Fr, I_te));
 %          te_1 = sum(Ytrain(Fr, I_te)) ./ sum(Ytrain(Fr, I_te) > 0);
 %        te_2 = mean(Ytrain(Fr2, I_te));
         te_1 = (S(1:N) * Ytrain(I(1:N), I_te)) / sum(S(1:N)); %./ (S(1:N) * Ypos(I(1:N), I_te));
%         te_2(abs((S(1:N) * Ypos(I(1:N), I_te))) < 1e-9) = 0;
        
         alpha = 1;
         
        TrainPredicted(i, I_tr) = tr_1 * alpha;% + tr_2 * (1 - alpha);
        TestPredicted(i, I_te) = te_1 * alpha;% + te_2 * (1-alpha);
 
   
   end    
    
    TrainPredicted(isnan(TrainPredicted)) = 0;
    TestPredicted(isnan(TestPredicted)) = 0;
    TrainPredicted(isinf(TrainPredicted)) = 0;
    TestPredicted(isinf(TestPredicted)) = 0;
    
    P_tr_0 = addBaseline(meanAll, meanU, meanI, TrainPredicted, I_tr_all);
    P_te_0 = addBaseline(meanAll, meanU, meanI, TestPredicted, I_te_all);
    
    TrainPredicted = P_tr_0;
    TestPredicted = P_te_0;
end

