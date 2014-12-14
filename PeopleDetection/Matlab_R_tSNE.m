%% Reading data - 1

clear all;
clc;
load 'Groups_2/train_test_feats_perplex_30_group_1_of_2_tSNE'
load 'Groups_2/train_labels_groups'

Tr = length(labels_groups{1});
Te = length(feats_tr_te_tSNE_group(:,1)) - Tr;
labels_1 = [labels_groups{1}; repmat(0, Te, 1)];

csvwrite('input_1.mat', feats_tr_te_tSNE_group);
csvwrite('label_1.mat', labels_1);

%% Reading data - 2
    
clear all;
clc;
load 'Groups_2/train_test_feats_perplex_30_group_2_of_2_tSNE'
load 'Groups_2/train_labels_groups'

Tr = length(labels_groups{2});
Te = length(feats_tr_te_tSNE_group(:,1)) - Tr;
labels_2 = [labels_groups{2}; repmat(0, Te, 1)];

csvwrite('input_2.mat', feats_tr_te_tSNE_group);
csvwrite('label_2.mat', labels_2);


%% Read data back
mat = csvread('output.mat');

fold = mat(:,1);
pred = mat(:,2);
real = mat(:,3);

tprAvg = zeros();
for i=1:9
    disp(i);
    p = pred(find(fold == i));
    r = real(find(fold == i));
    [tprAtWP,auc,fpr,tpr] = fastROC(r == 1, p, 0); 
    tprAvg(i) = tprAtWP;
end
