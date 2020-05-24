function [Ynorm, Ymean] = normalizeRatings(Y, R)
[m, n] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    idx = find(R(i, :) == 1);
    if any(idx)
      Ymean(i) = mean(Y(i, idx));
      Ynorm(i, idx) = Y(i, idx) - Ymean(i);
    else
      Ymean(i) = 0;     
    endif
end

end


