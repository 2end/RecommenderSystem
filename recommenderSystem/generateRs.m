function generateRs(num_rows, num_cols)
  train_coef = 0.6;
  val_coef = 0.2;
  num_elems = num_rows * num_cols;
  
  num_train_elems = int32(num_elems * train_coef);
  R = zeros(num_rows, num_cols);
  R = R(:);
  for i=1:num_train_elems
    R(i) = 1;
  endfor;
  R = R(randperm(num_elems));
  R = reshape(R, num_rows, num_cols);
  
  Rval = R == 0;
  Rval = Rval(:);
  num_val_elems = int32(num_elems * val_coef);
  count = num_val_elems;
  for i=1:num_elems
    if (Rval(i) == 1)
      Rval(i) = 0;
      count = count - 1;
      if (count == 0)
        break;
      endif
    endif
  endfor
  test = sum(Rval == 1);
  Rval = reshape(Rval, num_rows, num_cols);
  
  Rtest = R == Rval;
  
 save("Rs.mat", "R", "Rval", "Rtest");
end