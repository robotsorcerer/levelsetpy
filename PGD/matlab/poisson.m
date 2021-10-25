%%PGD Code for Poisson problems
% D. Gonzalez , I. Alfaro , E. Cueto
% Universidad de Zaragoza
% AMB -I3A Dec 2015
%

% Lekan Molu, 10/10/2021. See page 14, PGD book
clear all; close all; clc;
%% VARIABLES
%
ndim = 2; nn = 40.* ones (ndim ,1); % # of Dimensions , # of Elements
num_max_iter = 15; % Max . # of summands for the approach
TOL = 1.0e-4; npg = 2; % Tolerance , Gauss Points
coor = cell (ndim ,1); % Coordinates in each direction
L0 = -1.* ones (ndim ,1);
L1 = ones (ndim ,1); % Geometry (min ,max coordinates )
for i1 =1: ndim
  coor {i1} = linspace (L0(i1),L1(i1),nn(i1 ));
end
%%ALLOCATION OF IMPORTANT MATRICES
%
Km = cell (ndim ,1); % " Stiffness " matrix \int dN dN dx , Eq. (2.10)
Fv = cell (ndim ,1); % R and S sought enrichment functions
Mm = cell (ndim ,1); % " Mass " matrix \int N N dx , Eq. (2.11)
V = cell (ndim ,1); % Source term in separated form
%% COMPUTING STIFFNESS AND MASS MATRICES ALONG EACH DIRECTION
%

for i1 =1: ndim
  [Km{i1}, Mm{i1}] = elemstiff (coor{i1});
end

%% SOURCE TERM IN SEPARATED FORM
%%Let us begin by a separable expression . Evaluation of Eq. (2.13)
Ch{1 ,1} = @(x) cos (2* pi*x); Ch {2 ,1} = @(y) sin (2* pi*y);
% Try this new source term by yourself by simply uncommenting next 2 lines !
% Ch {1 ,1} = @(x) x.*x; Ch {1 ,2} = @(x) -1.0+0.0* x;
% Ch {2 ,1} = @(y) 1.0+0.0* y; Ch {2 ,2} = @(y) y.*y;
for j1 =1: ndim
  for k1 =1: size (Ch ,2)
    V{j1 }(: , k1) = Ch{j1 ,k1 }( coor {j1 });
  end
  % Although in this case we have a closed - form expression for the source
  % term , in general we know its nodal values .
  V{j1} = Mm{j1 }*V{j1 };
end
%% BOUNDARY CONDITIONS

%
CC = cell (ndim ,1);
for i1 =1: ndim
  IndBcnode {i1} = [1 numel(coor{i1})];
end

for i1 =1: ndim
  CC{i1} = setxor(IndBcnode{i1}, [1: numel(coor{i1})])';
end
%%ENRICHMENT OF THE APPROXIMATION , LOOKING FOR R AND S
%
num_iter = 0; iter = zeros (1); Aprt = 0; Error_iter = 1.0;
while Error_iter >TOL && num_iter < num_max_iter
  num_iter = num_iter + 1; R0 = cell (ndim ,1);
  for i1 =1: ndim
    % Initial guess for R and S.
    % It works equally well by choosing something random .
    R0{i1} = ones(numel(coor{i1}),1);
    % We impose that initial guess for functions R and S verify
    % homogeneous essential boundary conditions .
    R0{i1}(IndBcnode{i1}) = 0;
  end
  %%ENRICHMENT STEP
  %
  [R, iter(num_iter)] = enrichment(Km ,Mm ,V, num_iter ,Fv ,R0 ,CC , TOL );
  for i1 =1: ndim
     Fv{i1 }(: , num_iter ) = R{i1 };
  end % R (S) is valid , add it
  %%STOPPING CRITERION
  %
  Error_iter = 1.0;
  % One possible criterion is to stop when the norm of the new sum is
  % negligible wrt the pair of functions with the maximum norm
  for i1 =1: ndim
    Error_iter = Error_iter .* norm (Fv{i1 }(: , num_iter ));
  end

  Aprt = max (Aprt , sqrt ( Error_iter ));
  Error_iter = sqrt ( Error_iter )/ Aprt ;

  fprintf (1, '%dst   summand  in %d  iterations   with  a  weight  of %f\n' ,...
      num_iter , iter ( num_iter ), sqrt ( Error_iter ));
end

num_iter = num_iter - 1;% the last sum was negligible , we discard it.
fprintf(1, 'PGD   Process   exited   normally \n\n');
save ('WorkSpacePGD_Basic.mat');

%% POST - PROCESSING
%
for i1 =1: ndim
  figure ;
  plot ( coor {i1},Fv{i1 }(: ,1: num_iter ));
end

figure ;
if ndim ==2
  surf ( coor{1} , coor{2} , Fv{2}* Fv{1}');
end
