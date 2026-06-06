function dxdt = fx_model2(x,u,P,M)
% State equation
% x - states
% u - inputs
% P - parameters
% M - model specification

% Parameters as implcit 2x2 matrix, with cols = "from" and rows = "to",
% eg P = [1 2 3 4]; reshape(P,[2 2]) = 
%        [1 3
%         2 4]
% ie Cog decline = 1, Brain->Cog = 2; Cog->Brain = 3; Brain decline = 4

% Unpack states
BRAIN   = 1;
COGNITION = 2;
nstates = 2;

% Unpack parameters
P(1) = exp(P(1)) * -0.01; % BRAIN->BRAIN (ageing)
P(2) = exp(P(2)) * 0.01; % BRAIN->COGNITION
P(3) = exp(P(3)) * 0.01; % COGNITION->BRAIN
P(4) = exp(P(4)) * -0.01; % COGNITION->COGNITION

% State equations
dxdt = zeros(nstates,1);
dxdt(BRAIN)     = P(1) + P(3) * x(COGNITION);
dxdt(COGNITION) = P(4) + P(2) * x(BRAIN);