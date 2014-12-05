 function [ Ytest ] = DeratingTransform(Ytest, vals)
    if (~isempty(Ytest)) 
        vals = sort([0 vals +1e+9]);
        Ytest(find(Ytest < 0)) = 0;
        Ytest(find(Ytest > 1)) = 1;
        [Ytest, order_rev] = sort(Ytest);
        order = order_rev;
        order(order_rev) = 1:length(order_rev);
        
        ptr = 1;
        sumTot = sum(vals);
        sumCum = 0;
        trans = cumsum(vals) / sum(vals);
        
        for i=1:length(Ytest)
           while(ptr <= length(trans) && trans(ptr) <= Ytest(i)) 
               sumCum = sumCum + vals(ptr);
               ptr = ptr + 1;
           end
           dleft = Ytest(i) - trans(ptr - 1);
           dright = trans(ptr) - Ytest(i);
           Ytest(i) = (vals(ptr - 1) * dright + vals(ptr) * dleft) / (dleft + dright);
        end
        Ytest = Ytest(order);    
    end
end

