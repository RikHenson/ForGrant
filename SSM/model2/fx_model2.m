function dxdt = fx_model2(x,u,P,M)
% State equation
% x - states
% u - inputs
% P - parameters
% M - model specification

% Parameters:
% 1 = Brain decline (alpha_1)
% 2 = Cognitive decline (alpha_2)
% 3 = Brain->Cog (beta_1)
% 4 = Cog->Brain = 3 (beta_2)
%
% (Could reorder as implcit 2x2 matrix, with cols = "from" and rows = "to",
% eg P = [1 2 3 4]; reshape(P,[2 2]) = 
%        [1 3
%         2 4]

% Unpack parameters
P(1) = exp(P(1)) * -0.01; % BRAIN->BRAIN (ageing)
P(2) = exp(P(2)) * -0.01; % COGNITION->COGNITION

P(3) = exp(P(3)) * 0.01; % BRAIN->COGNITION
P(4) = exp(P(4)) * 0.01; % COGNITION->BRAIN

% State equations
BRAIN   = 1;
COGNITION = 2;
nstates = 2;
dxdt = zeros(nstates,1);
dxdt(BRAIN)     = P(1) + P(4) * x(COGNITION);
dxdt(COGNITION) = P(2) + P(3) * x(BRAIN);