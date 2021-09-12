% Inputs: 1) array-> input signal
%         2) dropped_tot-> minimum amount the signal must have dropped before
%         being considered a minimum
%         3) window_size-> amount of consecutive values to check before
%         exiting loop (must have also met the dropped_tot criteria)
%		  4) epsilon -> minimum drop in the change in the signal
%
% Output: 1) after_drop_ind-> minimum value subject to input constraints
function after_drop_ind = mymin(array_of_vals, dropped_tot, window_size,epsilon)
    min_val = array_of_vals(1);
    max_val = array_of_vals(1);
    not_min_count = 0;
    max_drop = 0;
    after_drop_ind = 1;
    
    for i=2:size(array_of_vals,2)
        if array_of_vals(i)<=min_val-epsilon
            not_min_count = 0;
            min_val = array_of_vals(i);   
            max_drop = max_val-min_val;
            after_drop_ind = i;
        elseif not_min_count >= window_size && abs(max_drop) > dropped_tot
            break
        elseif array_of_vals(i)>max_val
            max_val = array_of_vals(i);
            max_drop = max_val-min_val;
            not_min_count = not_min_count+1;
        else
            not_min_count = not_min_count+1;
        end      
    end
end