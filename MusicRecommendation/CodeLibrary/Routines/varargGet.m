% Recursively parse varargin cell array to find required pair
function [ ret, vars, found ] = varargGet( name, vararg )
    % Simplyfing if our varargin is nested
    vars = vararg;
 
    ret = 0;
    found = 0;
    
    if (~isempty(vars))
        for c=1:length(vars)
            if (iscell(vars{c}))
                [ret, tmp, found] = varargGet(name, vars{c});
                if (found > 0)
                    vars{c} = tmp;
                    return;
                end
            elseif ischar(vars{c})
                switch vars{c}
                   case {name}
                        ret = vars{c+1};
                        found = 1;
                        vars{c} = {};
                        vars{c + 1} = {};
                        return;
                    otherwise
                end % switch
            end
        end % for
    end
end

