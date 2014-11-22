% Recursively parse varargin cell array to find 
% required pair 'Name', Value.
% Function returns a modified version of vararg
% where the value that was extracted is cleared in the array
% The purpose of this is to prevent such a situation (e.x.)
% Parameter is passed to the CV function
% it calls Optimize function
% It itself calls CV function, with parameters added for CV function
% Problem: We have two records about same parameter in the varargin
function [ ret, vars, found ] = varargGet( name, vararg )
    vars = vararg;
 
    ret = 0;
    found = 0;
    
    if (~isempty(vars))
        for c=1:length(vars)
            % If we have a nested varargin, recursively to find in cell
            % This can happen when varargin from a previous function is
            % passed to a next function
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

