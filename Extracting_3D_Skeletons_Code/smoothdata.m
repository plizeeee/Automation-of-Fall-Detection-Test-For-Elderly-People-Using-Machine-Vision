function arr_new = smoothdata(arr, str, window_size)
    if str == 'movmedian'
       a= size(arr,1)
       for i = 1:length(arr)
          wind_start = i-floor(window_size/2);
          wind_end = i+floor(window_size/2);
          if wind_start<1
              wind_start = 1;
          end
          if wind_end>length(arr)
              wind_end = length(arr);
          end
          arr_new(i) = median(arr(wind_start:wind_end));
        end
    end
end
