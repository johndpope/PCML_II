%% Questions

% 1). What is the test set for the weak generalization
% 2). Is RMSE computed only on non-zero listener counts?

%% Data analysis and cleaning

clear all;
clc;
load 'Data/songTrain.mat'
G = Gtrain;
Y = Ytrain;
N = size(G, 1);

%% Distribution of listen counts for random users subset

x_d = 5;
y_d = 5;
idx = randsample(N, x_d * y_d);
figure;
for i=1:length(idx)
    vals = Y(idx(i),find(Y(idx(i),:) > 0));
    vals = sort(vals, 'descend');
    subplot(x_d, y_d, i);
    plot(vals);
end

%% Minimum number of songs per user and users per song
T = boolean(Y);
bysong = sum(T, 1);
byuser = sum(T, 2);

%% Statistics to be used in the report
stat = full(sum(Y, 1));
stat_b = full(sum(boolean(Y), 1));
fprintf('Statistics about how many times artists were listened\n');
fprintf('Number of songs never played: %d\n', sum(stat == 0));
fprintf('Maximum number of listen counts per artist: %d, for artist %s\n', max(stat), artistName{find(stat == max(stat))});
fprintf('Maximum number of user counts per artist: %d, for artist %s\n', max(stat_b), artistName{find(stat_b == max(stat_b))});
fprintf('Mean number of listen counts per artist: %0.4f\n', mean(stat));
fprintf('Mean number of user counts per artist: %0.4f\n', mean(stat_b));
fprintf('Median number of listen counts per artist: %0.1f\n', median(stat));
fprintf('Median number of user counts per artist: %0.1f\n', median(stat_b));
tmp = Y(:);
fprintf('Total average listen count: %0.1f\n', mean(full(tmp(tmp > 0))));


stat = full(sum(Y, 2));
stat_b = full(sum(boolean(Y), 2));
fprintf('Statistics about how many artists users listened to\n');
fprintf('Number of users who didn t listen to anything: %d\n', sum(stat == 0));
fprintf('Maximum number of listen counts per user: %d\n', max(stat));
fprintf('Maximum number of artist counts per user: %d\n', max(stat_b));
fprintf('Mean number of listen counts per user: %0.4f\n', mean(stat));
fprintf('Mean number of artist counts per user: %0.4f\n', mean(stat_b));
fprintf('Median number of listen counts per user: %0.1f\n', median(stat));
fprintf('Median number of artist counts per user: %0.1f\n', median(stat_b));


stat = full(sum(G, 1));
fprintf('Statistics about friendship graphs\n');
fprintf('Number of users with no friends: %d\n', sum(stat == 0));
fprintf('Maximum number of friends per user: %d\n', max(stat));
fprintf('Mean number of friends per user: %0.4f\n', mean(stat));
fprintf('Median number of friends per user: %d\n', median(stat));


fprintf('Correlation between number of friends and listen counts: %0.4f\n'...
    , corr(full(sum(G,1))', full(sum(Y, 2)))); 
fprintf('Correlation between number of artist listened and listen counts: %0.4f\n'...
    , corr(full(sum(boolean(Y),2)), full(sum(Y, 2)))); 

