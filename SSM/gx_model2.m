function y = gx_model2(x,u,P,M)
% Observation model
% x - states
% u - inputs
% P - parameters
% M - model specification

% Unpack states
BRAIN   = 1;
COGNITION = 2;

% Unpack parameters
nout = 2;
y = zeros(1,nout);
y(BRAIN) = P(5) + x(BRAIN); 
y(COGNITION) = P(6) + x(COGNITION); 