function [p1 ,p2] = elemstiff ( coor )
  % function [p1 ,p2] = elemstiff ( coor )
  % For the coordinates coor , obtains p1 ( Stiffness ) and p2( Mass ) matrices
  % Universidad de Zaragoza - 2015
  nen = numel ( coor ); p1 = zeros (nen ); p2 = zeros (nen ); % p3 = zeros (nen ,1);
  X = coor(1: nen -1)'; % Left coordinate of the elements
  Y = coor(2: nen )'; % Right coordinate of the elements
  L = Y - X; % Length of the elements
  sg = [ -0.57735027 , 0.57735027]; 
  wg = [1, 1]; % Gauss and weight points
  npg = numel (sg );
  for i1 =1: nen -1
    c = zeros (1, npg );
    N = zeros (nen ,npg );
    dN = zeros (nen ,npg );
    c(1 ,:) = 0.5.*(1.0 - sg ).*X(i1) + 0.5.*(1.0+ sg ).*Y(i1 );
    N(i1 +1 ,:) = (c(1 ,:) -X(i1 ))./ L(i1 );
    N(i1 ,:) = (Y(i1)-c (1 ,:))./ L(i1 );
    dN(i1 +1 ,:) = ones (1, npg )./ L(i1 );
    dN(i1 ,:) = -dN(i1 +1 ,:);
    for j1 =1: npg
      p1 = p1 + dN (:, j1 )* dN (:, j1 )'*0.5.* wg(j1 ).*L(i1 ); % dNudN
      p2 = p2 + N(:, j1 )*N(:, j1 )'.*0.5.* wg(j1 ).*L(i1 ); % NuN
    end
  end
return
