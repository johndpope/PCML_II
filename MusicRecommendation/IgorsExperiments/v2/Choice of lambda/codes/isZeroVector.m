function answer = isZeroVector(arr)
    N = length(arr);
    for k=1:N
        if (arr(k) ~= 0)
            answer = 0;
            break;
        end
    end
    answer = 1;
end
