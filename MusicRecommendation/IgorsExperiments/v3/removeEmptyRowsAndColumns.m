function [Ytr_1, Gtr_1] = removeEmptyRowsAndColumns(Ytr_0, Itr_0, Gtr_0)

    numOfUsers = size(Ytr_0, 1);
    numOfItems = size(Ytr_0, 2);
    
    nonEmptyRows = zeros(numOfUsers, 1);
    for i=1:numOfUsers
        if (sum(sum(Itr_0(i, :))) > 0)
            nonEmptyRows(i) = 1;
        end
    end
    
    nonEmptyColumns = zeros(numOfItems, 1);
    for j=1:numOfItems
        if (sum(sum(Itr_0(:, j))) > 0)
            nonEmptyColumns(j) = 1;
        end
    end
    
    Ytr_1 = Ytr_0(find(nonEmptyRows == 1), find(nonEmptyColumns == 1));
    Gtr_1 = Gtr_0(find(nonEmptyRows == 1), find(nonEmptyRows == 1));
end
