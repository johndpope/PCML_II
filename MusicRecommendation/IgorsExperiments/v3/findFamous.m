function famous = findFamous(Ytr, Itr, threshold)
    numOfItems = size(Ytr, 2);
    averageNumberOfCounts = zeros(numOfItems, 1);
    famous = zeros(numOfItems, 1);
    for j=1:numOfItems
        if (sum(sum(Itr(:, j))) == 0)
            averageNumberOfCounts(j) = -1;
        else
            averageNumberOfCounts(j) = sum(sum(Ytr(:, j))) / sum(sum(Itr(:, j)));
            if (averageNumberOfCounts(j) > threshold)
                famous(j) = 1;
            end
        end
    end
end
