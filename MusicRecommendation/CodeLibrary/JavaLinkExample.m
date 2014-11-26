% Example of linking to make call to Java
clear all;
clc;
load('Data/songTrain.mat');

N = size(Ytrain, 1);
K = size(Ytrain, 2);

tN = 200;
tM = 220;

Y_tr = sparse(N, K);
Y_te = sparse(N, K);
G_tr = sparse(N, N);
G_tr_te = sparse(N, N);
G_te_tr = sparse(N, N);
G_te = sparse(N, N);

Y_tr(1:tN,:) = Ytrain(1:tN,:);
Y_te((tN + 1):tM,:) = Ytrain((tN + 1):tM,:);
G_tr(1:tN, 1:tN) = Gtrain(1:tN, 1:tN);
G_tr_te(1:tN, (tN + 1):tM) = Gtrain(1:tN, (tN + 1):tM);
G_tr_te((tN + 1):tM, 1:tN) = Gtrain((tN + 1):tM, 1:tN);
G_te((tN + 1):tM, (tN + 1):tM) = Gtrain((tN + 1):tM, (tN + 1):tM);
%% See how to call Java:
% example of use: Constant_Java_TrainAndPredict_Strong

[TrainPredicted1, TestPredicted1] = Constant_TrainAndPredict_Strong(...
    G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te);

[TrainPredicted2, TestPredicted2] = Constant_Java_TrainAndPredict_Strong(...
    G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te);

disp([mean(TrainPredicted1(TrainPredicted1 > 0))...
      mean(TrainPredicted2(TrainPredicted2 > 0))]);
disp([mean(TestPredicted1(TestPredicted1 > 0))...
      mean(TestPredicted2(TestPredicted2 > 0))]);
  
% Now same thing, with cross-validation - here results are the same
[TrainError1, TestError1] = crossValidation_Strong(...
  G_tr, Y_tr, @Constant_TrainAndPredict_Strong, ...
  'CV_type', 'CV', 'CV_k', 10, 'CV_verbose', 1);

[TrainError2, TestError2] = crossValidation_Strong(...
  G_tr, Y_tr, @Constant_Java_TrainAndPredict_Strong, ...
  'CV_type', 'CV', 'CV_k', 10, 'CV_verbose', 2);


%% Now same check for per artist average

[TrainPredicted1, TestPredicted1] =...
    ConstantPerArtist_TrainAndPredict_Strong(...
    G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te);

[TrainPredicted2, TestPredicted2] = ...
    ConstantPerArtist_Java_TrainAndPredict_Strong(...
    G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te);

disp([mean(TrainPredicted1(TrainPredicted1 > 0))...
      mean(TrainPredicted2(TrainPredicted2 > 0))]);
disp([mean(TestPredicted1(TestPredicted1 > 0))...
      mean(TestPredicted2(TestPredicted2 > 0))]);
  
  
 % Now same thing, with cross-validation - here results are the same
[TrainError2, TestError2] = crossValidation_Strong(...
  G_tr, Y_tr, @ConstantPerArtist_Java_TrainAndPredict_Strong, ...
  'CV_type', 'CV', 'CV_k', 10, 'CV_verbose', 2);

[TrainError1, TestError1] = crossValidation_Strong(...
  G_tr, Y_tr, @ConstantPerArtist_TrainAndPredict_Strong, ...
  'CV_type', 'CV', 'CV_k', 10, 'CV_verbose', 1);

