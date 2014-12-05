function [ Ytest ] = ratingTransform(Ytest, vals)
    vals = sort([0 vals +1e+9]);
    [Ytest, order_rev] = sort(Ytest);
    order = order_rev;
    order(order_rev) = 1:length(order_rev);
    ptr = 1;
    trans = cumsum(vals) / sum(vals);
    
    for i=1:length(Ytest)
       while(ptr <= length(vals) && vals(ptr) <= Ytest(i)) 
           ptr = ptr + 1;
       end
       dleft = Ytest(i) - vals(ptr - 1);
       dright = vals(ptr) - Ytest(i);
       Ytest(i) = (trans(ptr - 1) * dright + trans(ptr) * dleft) / (dleft + dright);
    end
    Ytest = Ytest(order);
end

