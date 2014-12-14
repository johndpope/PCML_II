function X = transformFeats(feats)

% Transform the input features
fprintf('Feature transformation...'); tic

% Input parameters
num_feats = length(feats);

% Make the image descriptor a columns wih 9360 dimensions
num_dim = numel(feats{1});
X = zeros(num_feats, num_dim, 'single');
for i = 1:num_feats
    X(i, :) = feats{i}(:);
end

% Print the elapsed time
fprintf(' %.4f seconds\n', toc);

% ///////////////////// Other feature transformations /////////////////////

% Average the different normalizations
% num_dim = numel(feats{1}) / 4;
% feats_mat = zeros(num_feats, num_dim, 'single');
% for i = 1:num_feats
%     for j = 0:3
%         feats_mat(i, :) = feats_mat(i, :) + feats{i}((1:num_dim) + j * num_dim)';
%     end
% end


% Sum all the image features in the third dimension, i.e. sum all HOGs from
% the image to have a single value (HOG energy)
% feats_mat = cell(num_feats, 1);
% for i = 1:num_feats
%     feats_mat{i} = sum(feats{i}, 3); % Sum
% %     for j = 1:size(feats{i}, 1)    % Norm
% %         for k = 1:size(feats{i}, 2)
% %             feats_mat{i}(j, k) = norm(reshape(feats{i}(j, k, :), [size(feats{i}, 3), 1]));
% %         end
% %     end
% end


% Keep only the maximal element for each HOG (maximal from any of the 4 normalizations)
% feats_mat = cell(num_feats, 1);
% elements_per_image = size(feats{1}, 1) * size(feats{1}, 2);
% max_hog_image = reshape(1:elements_per_image, [size(feats{1}, 1), size(feats{1}, 2)]);
% for i = 1:num_feats
%     [val, idx] = max(feats{i}, [], 3);
%     v = zeros(size(feats{i}), 'single');
%     v((idx(:) - 1) * elements_per_image + max_hog_image(:)) = val(:); % keep only the maximal value
%     feats_mat{i} = v;
% end

% Compute the mean of each histogram orientation, i.e. each image has 36 dimensions
% num_dim = size(feats{1}, 3);
% X = zeros(num_feats, num_dim, 'single');
% for i = 1:num_feats
%     X(i, :) = mean(reshape(feats{i}(:), [size(feats{1}, 1) * size(feats{1}, 2), size(feats{1}, 3)]));
% end
% 
% pedestrian_idx = find(labels > 0);
% non_pedestrian_idx = find(labels <= 0);
% pedestrian_hog_mean = mean(X(pedestrian_idx, :));
% non_pedestrian_hog_mean = mean(X(non_pedestrian_idx, :));
% figure, plot(1:36, [non_pedestrian_hog_mean; pedestrian_hog_mean; X(non_pedestrian_idx(round(rand(1) * 1200)), :)]); axis([1 36 0 0.2]); legend('Neg. mean', 'Pos. mean', 'Neg. sample');
% figure, plot(1:36, [non_pedestrian_hog_mean; pedestrian_hog_mean; X(pedestrian_idx(round(rand(1) * 1200)), :)]); axis([1 36 0 0.2]); legend('Neg. mean', 'Pos. mean', 'Pos. sample');
% 
% pedestrian_hog_mean = mean(X(pedestrian_idx, :));
% non_pedestrian_hog_mean = mean(X(non_pedestrian_idx, :));
% pos_correlation = zeros(num_feats, 1);
% neg_correlation = zeros(num_feats, 1);
% normalization = 0.2 * 0.2 * 36;
% for i = 1:num_feats
%     pos_correlation(i) = X(i, :) * pedestrian_hog_mean' / normalization;
%     neg_correlation(i) = X(i, :) * non_pedestrian_hog_mean' / normalization;
% end
% 
% figure, plot([pos_correlation(pedestrian_idx)', pos_correlation(non_pedestrian_idx)']);
% figure, plot([abs((pos_correlation(pedestrian_idx)' ./ neg_correlation(pedestrian_idx)') - mean(pos_correlation(pedestrian_idx) ./ neg_correlation(pedestrian_idx))),...
%       abs((pos_correlation(non_pedestrian_idx)' ./ neg_correlation(non_pedestrian_idx)') - mean(pos_correlation(pedestrian_idx) ./ neg_correlation(pedestrian_idx)))]);

end