function nre = getNumberOfNonZeros(M)
    nre = length(find(M > 0));
end
