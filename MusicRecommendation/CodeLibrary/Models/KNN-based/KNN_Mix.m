function [ TrainPredicted, TestPredicted ] = ...
    KNN_Mix(...
    Ytrain,...      % Matrix of user - artist listen counts in train    
    Ytest,...       % Indices, for which we are insterested in answer
    G,Itr,Ite,splt,...             %
    varargin...     %
)

% By users

    %splt = 15;
%     HI = find(sum(Ytrain > 0, 2) > splt);
%     LI = find(sum(Ytrain > 0, 2) <= splt);
%     
% %   fprintf('Split: HI = %d, LI = %d\n', length(HI), length(LI));
%     
%     [A1, B1] = KNN_Friend_TrainAndPredict(...
%         Ytrain(HI,:), Ytest(HI,:), G(HI,HI),...
%         Itr(HI,:), Ite(HI,:));
%     
%     [A2, B2] = KNN_Friend_TrainAndPredict(...
%         Ytrain(LI,:), Ytest(LI,:), G(LI,LI),...
%         Itr(LI,:), Ite(LI,:));
% 
% %     [A1, B1] = stackAlgorithm(...
% %         Ytrain(HI,:), Ytest(HI,:),...
% %         Itr(HI,:), Ite(HI,:));
% %     
% %     [A2, B2] = stackAlgorithm(...
% %         Ytrain(LI,:), Ytest(LI,:),...
% %         Itr(LI,:), Ite(LI,:));
% 
%     TrainPredicted = Ytrain;
%     TrainPredicted(HI,:) = A1;
%     TrainPredicted(LI,:) = A2;
%     
%     TestPredicted = Ytest;
%     TestPredicted(HI,:) = B1;
%     TestPredicted(LI,:) = B2;



    HI = find(sum(Ytrain > 0, 1) > splt);
    LI = find(sum(Ytrain > 0, 1) <= splt);
    
   fprintf('Split: HI = %d, LI = %d\n', length(HI), length(LI));
    
    [A1, B1] = KNN_Friend_TrainAndPredict(...
        Ytrain(:,HI), Ytest(:,HI), G,...
        Itr(:,HI), Ite(:,HI));
    
    [A2, B2] = KNN_Friend_TrainAndPredict(...
        Ytrain(:,LI), Ytest(:,LI), G,...
        Itr(:,LI), Ite(:,LI));

%     [A1, B1] = stackAlgorithm(...
%         Ytrain(HI,:), Ytest(HI,:),...
%         Itr(HI,:), Ite(HI,:));
%     
%     [A2, B2] = stackAlgorithm(...
%         Ytrain(LI,:), Ytest(LI,:),...
%         Itr(LI,:), Ite(LI,:));

    TrainPredicted = Ytrain;
    TrainPredicted(:,HI) = A1;
    TrainPredicted(:,LI) = A2;
    
    TestPredicted = Ytest;
    TestPredicted(:,HI) = B1;
    TestPredicted(:,LI) = B2;



    
    TrainPredicted(find(TrainPredicted < 0)) = 0;
    TestPredicted(find(TestPredicted < 0)) = 0;
end

