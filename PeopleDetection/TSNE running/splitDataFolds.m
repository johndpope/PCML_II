function [data, idxCV, y] = splitDataFolds(X, K, seed, labels)

N = size(X, 1);
%setSeed(seed)
%idx = randperm(N);
idx = 1:N;
Nk = floor(N/K);
limits = [(0:(K - 1)) * Nk, N] + 1; % the +1 is just because of the loop where we have -1
distribute_workload = [0, 1:mod(Nk, K),...
    ones(1, K - mod(Nk, K) - 1) * mod(N, K), 0];
limits = limits + distribute_workload;

idxCV = cell(K, 1);
data = cell(K, 1);
y = cell(K, 1);
for k = 1:K
    idxCV{k} = idx(limits(k):limits(k + 1) - 1);
    data{k} = X(idxCV{k}, :);
    if exist('labels', 'var'), y{k} = labels(idxCV{k}); end
end
end

% seed = 2143 provides good positive-negative balancing
% fprintf('Total: %.4f | %.4f | %.4f | %.4f | %.4f\n', sum(labels > 0) / sum(labels < 0),...
% sum(labels(idxCV{1}) > 0) / sum(labels(idxCV{1}) < 0),...
% sum(labels(idxCV{2}) > 0) / sum(labels(idxCV{2}) < 0),...
% sum(labels(idxCV{3}) > 0) / sum(labels(idxCV{3}) < 0),...
% sum(labels(idxCV{4}) > 0) / sum(labels(idxCV{4}) < 0));