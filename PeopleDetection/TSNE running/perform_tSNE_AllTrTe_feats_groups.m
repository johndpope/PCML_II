clearvars;
load('Data/train_feats'); % CHANGE TO YOUR DATA FOLDER

fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X = zeros([length(feats) D]);

for i=1:length(feats)
    X(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end

load('Data/test_feats'); % CHANGE TO YOUR DATA FOLDER

fprintf('Generating feature vectors..\n');
D = numel(feats{1});  % feature dimensionality
X2 = zeros([length(feats) D]);

for i=1:length(feats)
    X2(i,:) = feats{i}(:);  % convert to a vector of D dimensions
end



feats_tr_te_tSNE_group = tsne([X(1:500,:); X2(1:500,:)],...
    [], 2, 30);


figure;
lab = (labels_groups{1});
t = feats_tr_te_tSNE_group(1:length(lab),:);
plot(t(lab == 1,1), t(lab == 1,2),'.r');
hold on;
plot(t(lab == -1,1), t(lab == -1,2),'.b');

