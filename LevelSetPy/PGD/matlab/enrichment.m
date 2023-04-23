function [R, iter ] = enrichment (K,M,V, num_iter ,FV ,R,CC , TOL)
  % function [R, iter ] = enrichment (K,M,V, num_iter ,FV ,R,CC , TOL)
  % Compute a new sumand by fixed - point algorithm using PGD
  % Universidad de Zaragoza , 2015
  iter = 1;
  mxit = 25; % # of possible iterations for the fixed point algorithm
  error = 1.0e8; % Initialization
  ndim = size (FV ,1); % Number of Variables

  %%  FIXED POINT ALGORITHM
  %
  while abs( error )>TOL
  Raux = R; % Remember : R is a cell containing both R and S
  for i1 =1: ndim % Alternating between R and S
    matrix = zeros ( numel (R{i1 })); % K matrix in Eq. (2.14)
    source = zeros ( size (R{i1 } ,1) ,1);% V2 -V1 in Eq. (2.14)
    %%  COMPUTING K MATRIX Eq (2.14)
    %
    for i2 =1: ndim % Products in sum = ndim (2 in this case )
      FTE = 1.0; % Computing F^T E in Eq. (2.8)
      % Remember : K = \int dN dN dx and M = \int N N dx
      for i3 =1: ndim
        if i3 == i2
          if i3 == i1
            FTE = FTE .*K{i3 };
          else
            FTE = FTE .*(R{i3 }'*K{i3 }*R{i3 });
          end
        else
          if i3 == i1
            FTE = FTE .*M{i3 };
          else
            FTE = FTE .*(R{i3 }'*M{i3 }*R{i3 });
          end
        end
      end
      matrix = matrix + FTE ;
    end
  %%  COMPUTING V2 in Eq. (2.14)
  %
  for j1 =1: size (V{i1 } ,2) % Number functions of the source
    V2 = 1.0;
    for i2 =1: ndim
      if i2 == i1
        V2 = V2 .*V{i2 }(: , j1 );
      else
        V2 = V2.*(R{i2}'*V{i2 }(: , j1 ));
      end
    end
    source = source + V2;
  end
  %%  COMPUTING V1 in Eq. (2.14)
  %
  for j1 =1: num_iter -1
    for i2 =1: ndim % Terms in sum
      FTD = 1.0; % COMPUTING F^T D in Eq. (2.9)
      for i3 =1: ndim
      if i3 == i2
        if i3 == i1
          FTD = FTD .*(K{i3 }* FV{i3 }(: , j1 ));
        else
          FTD = FTD .*(R{i3 }'*K{i3 }* FV{i3 }(: , j1 ));
        end
      else
        if i3 == i1
          FTD = FTD .*(M{i3 }* FV{i3 }(: , j1 ));
        else
          FTD = FTD .*(R{i3 }'*M{i3 }* FV{i3 }(: , j1 ));
        end
      end
    end
    source = source - FTD ; % Note that source = V2 -V1
    end

  end
  %%  SOLVE Eq. (2.14) FOR EACH DIRECTION
  R{i1}(CC{i1}) = matrix (CC{i1},CC{i1 })\source (CC{i1 });
  % We normalize S. R takes care of the alpha constant in Eq. (2.2)
  if i1 ~=1  R{i1} = R{i1 }./ norm (R{i1 }); end
  end
  % If two successive Rs are too similar , we stop
  error = 0;
  for j1 =1: ndim
    error = error + norm ( Raux {j1}-R{j1 });
  end
  error = sqrt ( error );
  iter = iter + 1;
  if iter == mxit % If we reach the max # of iterations , we exit
  return ;
  end
  end
return
