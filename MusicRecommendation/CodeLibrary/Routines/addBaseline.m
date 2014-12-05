function [ Y_te ] = addBaseline(meanAll, meanU, meanI, Y_te, I_te)
    Y_te(I_te) = Y_te(I_te) + meanAll;
    
    for i=1:size(Y_te, 1)
        idx = find(I_te(i,:) > 0);
        Y_te(i,idx) = Y_te(i,idx) + meanU(i);
    end
    
    for i=1:size(Y_te, 2)
        idx = find(I_te(:,i) > 0);
        Y_te(idx,i) = Y_te(idx,i) + meanI(i);
    end
end

