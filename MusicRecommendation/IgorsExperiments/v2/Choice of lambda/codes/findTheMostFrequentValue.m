function [value, longest] = findTheMostFrequentValue(arr)
    arrS = sort(arr);
    T = length(arr);
    
    longest = -1;
    value = -1;
    
    previous = log(0);
    consecutive = 0;
    for i=1:T
        if (arrS(i) == previous)
            consecutive = consecutive + 1;
            if (consecutive > longest)
                longest = consecutive;
                value = arrS(i);
            end            
        else
            consecutive = 1;
            previous = arrS(i);
            if (consecutive > longest)
                longest = consecutive;
                value = arrS(i);
            end
        end
    end
    
end

