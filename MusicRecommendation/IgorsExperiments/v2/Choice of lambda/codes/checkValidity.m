function checkValidity(Y)
    numOfUsers = size(Y, 1);
    numOfItems = size(Y, 2);
    
    everythingGood = 1;
    
    for i=1:numOfUsers
        ss1 = length(find(Y(i, :) > 0));
        if (ss1 == 0)
            fprintf('Problem with user %d.\n.', i);
            everythingGood = 0;
        end
    end
    
    for j=1:numOfItems
        ss2 = length(find(Y(:, j) > 0));
        if (ss2 == 0)
            fprintf('Problem with item %d.\n', j);
            everythingGood = 0;
        end
    end
    
    if (everythingGood == 1)
        fprintf('The validity test was successfull!\n');
    end
    
end

