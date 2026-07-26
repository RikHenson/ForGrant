%% Demonstration of fitting a 2-population model to the OASIS dataset
%
% Rik Henson and Peter Zeidman

%% Import and prepare data
addpath('toolbox');
addpath('model2');

% Load data and unpack
load('data/OASIS_longitudinal_data.mat');

age = OASIS_longitudinal_data.age;
id  = OASIS_longitudinal_data.id;

% Get unique subject names
uids = unique(id);
nsubjects = length(uids);

% Zscore data
zscore_xnan = @(x) bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'omitnan')), std(x, 'omitnan'));

brain = zscore_xnan(OASIS_longitudinal_data.Cortical_Thickness);
cognition = zscore_xnan(OASIS_longitudinal_data.Fluid_Intelligence); % already Z-scored?

%% Remove one-trial practice effects?
% not_nan = find(~isnan(cognition));
% y = cognition(not_nan);
% id_data = id(not_nan);
% X = zeros(length(not_nan),nsubjects+1);
% for i = 1:nsubjects
%     ind = find(id_data == uids(i));
%     X(ind,1) = [0; ones(length(ind)-1,1)];
%     X(ind,i+1) = ones(size(ind));
% end
% B = pinv(X)*y;
% y = y - X*B;
% y = y + X(:,2:end)*B(2:end);
% cognition(not_nan) = y;

% Decide the number of evenly spaced age bins
min_age_in_years = floor(min(age));
max_age_in_years = ceil(max(age));
nbins = max_age_in_years - min_age_in_years + 1;
bin_centres = linspace(min_age_in_years, max_age_in_years, nbins);

% Assign each measurement to the nearest age bin
edges = linspace(min(age), max(age), nbins);
age_bin_idx = discretize(age, edges);

% Prepare subject-level data with nans for missing values
% (could make Y a 3D matrix, but keep brain and cog separate for clarity)
Y_brain = nan(nbins,nsubjects);
Y_cognition = nan(nbins,nsubjects);
for i = 1:nsubjects
    % Identify rows of the data for this subject
    rows = strcmp(id,uids{i});
    subject_age_bin_idx  = age_bin_idx(rows);

    % Store
    Y_brain(subject_age_bin_idx,i) = brain(rows);
    Y_cognition(subject_age_bin_idx,i) = cognition(rows);
end

% Remove all-nan subjects
to_retain = ~(all(isnan(Y_brain)) | all(isnan(Y_cognition)));
Y_cognition = Y_cognition(:,to_retain);
Y_brain = Y_brain(:,to_retain);
nsubjects   = sum(to_retain)

% Prepare grand mean data for initial fitting
Y_mean_brain = mean(Y_brain,2,"omitnan");
Y_mean_cognition = mean(Y_cognition,2,"omitnan");
N_mean_brain = sum(~isnan(Y_brain),2);
N_mean_cognition = sum(~isnan(Y_cognition),2);

%% Visualise data
%spm_figure('GetWin','Data');
%spm_clf;
figure(Position = [0 70 700 1000])

% Plot data present / absence
% subplot(3,1,1);
% any_data = ~isnan(Y_brain)';
% imagesc(edges,1:nsubjects,any_data);colormap gray;
% ylabel('Subject');xlabel('Age');title('Data: Cortical brain');
% set(gca,'FontSize',12);

% Plot cortical brain data
hold on
co = colororder;
for i = 1:nsubjects
    y = Y_brain(:,i);
    k = ~isnan(y);
    x = edges(k);
    y = y(k);
    plot(x,y,'.-','Color',co(1,:));

    y = Y_cognition(:,i);
    k = ~isnan(y);
    x = edges(k);
    y = y(k);
    plot(x,y,'.-','Color',co(2,:));
end
any_data = ~isnan(Y_mean_brain);
plot(edges(any_data),Y_mean_brain(any_data),'LineWidth',3,'Color',co(1,:));
any_data = ~isnan(Y_mean_cognition);
plot(edges(any_data),Y_mean_cognition(any_data),'LineWidth',3,'Color',co(2,:));
xlabel('Age'); ylabel('Z-scored Brain/Cognition')
legend({'Brain','Cognition'})
title('Data');
set(gca,'FontSize',12);
%ylim([2 2.8]);

%% Simulate data under priors

% Starting values for states 
x0 = [1 1]';

% Priors
%M.pE = [0 0 0 0 1 1]';
M.pE = [3/2 1 1/2 3/2 0 0]'; % chosen to resemble mean data roughly
M.pC = diag([1/16 1/16 1/16 1/16 1 1]);

% Model spec
M.IS = @ode_LL;
M.f  = @fx_model2;
M.g  = @gx_model2;
M.x  = x0; % starting values
M.l  = 2;  % number of outputs
M.m  = 1;  % number of inputs (columns of U.u) - does this need to match number of states/outputs?
M.ns = nbins; % number of samples

% Data structure
U = struct();
U.u = zeros(nbins,M.m);
U.dt = 1;

% Integrate
[y,ty,x,tx] = ode_LL(M.pE,M,U);

% Switch off brain<->cognition interactions and re-integrate
P = M.pE;
P(2) = -32; % brain->cognition
P(3) = -32; % cognition->brain
[ylinear,ty,xlinear,tx] = ode_LL(P,M,U);

% Create figure
%spm_figure('GetWin','Simulation under priors');
figure(Position = [0 0 700 1000])

% Plot states (no-interaction model)
subplot(2,2,1);
plot(edges,xlinear,'LineWidth',3);
legend({'Brain','Cognition'},'Location','west');
xlabel('Age');
title('Latent States');
axis square
set(gca,'FontSize',12);

% Plot predicted data (no-interaction model)
subplot(2,2,2);
plot(edges,ylinear,'LineWidth',3);
legend({'Brain','Cognition'},'Location','west');
xlabel('Age');
title('Simulated data');
axis square
set(gca,'FontSize',12);

% Plot states (full model)
subplot(2,2,3);
plot(edges,x,'LineWidth',3);
legend({'Brain','Cognition'},'Location','west');
xlabel('Age');
title('Latent States');
axis square
set(gca,'FontSize',12);

% Plot predicted data (full model)
subplot(2,2,4);
plot(edges,y,'LineWidth',3);
legend({'Brain','Cognition'},'Location','west');
xlabel('Age');
title('Simulated data');
axis square
set(gca,'FontSize',12);

%% Fit model to grand mean cortical brain

% Less biased priors?
M.pE = [0 0 0 0 0 0]'; % shrinkage priors
M.pC = diag([1/16 1/16 1/16 1/16 1 1]);
%M.pE = [3/2 1 1/2 3/2 0 0]'; % chosen to resemble mean data roughly
%M.pC = diag([1/16 1/16 1/16 1/16 64 64]);

% Noise priors
%M.hE = 6;
%M.hC = 1/1024;
M.hE = 6;
M.hC = 1/6;

% Measurement precision (nans have low prior precision)
% Q(i,i) = N where N is the number of measurements (with a min of 1/1024)
Q={};
%Q = {diag(max(1/1024,N_mean_brain+N_mean_cognition))};
%Q = spm_Ce(repmat(nbins,1,2)); % appropriate if observations concatenated
Q{1} = diag(max(1/1024,N_mean_brain));
Q{2} = diag(max(1/1024,N_mean_cognition));
% e = zeros(2); e(1,1)=1;
% Q{1} = kron(e,diag(max(1/1024,N_mean_brain)));
% e = zeros(2); e(2,2)=1;
% Q{2} = kron(e,diag(max(1/1024,N_mean_cognition)));

% Set data
Y = struct();
Y.y  = [Y_mean_brain Y_mean_cognition];
Y.y(isnan(Y.y(:,1)),1) = mean(Y.y(:,1),"omitnan");
Y.y(isnan(Y.y(:,2)),2) = mean(Y.y(:,2),"omitnan");
%Y.y = [Y.y(:,1); Y.y(:,2)];
Y.dt = 1;
Y.Q  = Q;

% Invert
%[Ep,Cp,Eh,Ch,F] = variational_laplace(M,U,Y); % does not handle >1 variable
M.X0 = ones(size(Y.y,1),1);
%M.X0 = kron(eye(2),ones(nbins,1));
[Ep,Cp,Eh,Ch,F] = spm_nlsi_GN(M,U,Y);

% Integrate under posteriors
[yhat,ty,xhat,tx] = ode_LL(Ep,M,U);

% Save in standard DCM format
DCM = struct();
DCM.Ep = Ep;
DCM.Cp = Cp;
DCM.Eh = Eh;
DCM.Ch = Ch;
DCM.F  = F;
DCM.M = M;
DCM.Y = Y;
DCM.U = U;
DCM.yhat = yhat;
DCM.ty   = ty;
DCM.xhat = xhat;
DCM.tx   = tx;
DCM.ages = edges;

% Bayesian model reduction: reduced model with/without each parameter

% Full priors
pE = M.pE;
pC = M.pC;

models = [
    1 0 0 1  1 1;  % No cross-terms
    1 0 1 1  1 1;  % No B->C
    1 1 0 1  1 1   % No C->B
]
nmodels = size(models,1)
Pp = zeros(nmodels,1);
for m = 1:nmodels
    % Reduced prior mean
    rE = pE;
    rP = find(~models(m,:));
    rE(rP) = -32;

    % Reduced prior covariance
    rC = pC;
    rC(rP,rP) = 0; % point null quite unlikely?
    %rC(rP,rP) = mean(pE)/100;

    % Compare models (reduced - full free energy)
    dF = spm_log_evidence_reduce(Ep,Cp,pE,pC,rE,rC);

    % F -> P
    P = spm_softmax([dF; 0]);
    Pp(m) = P(2);
%    fprintf('Model %d: Log bayes factor versus full model: %2.2f, P=%2.2f\n',...
%        m,dF,P(2));   
end
Pp

%Pp = 1 - spm_Ncdf(pE,abs(Ep),diag(Cp)) % different from priors
%Pp = 1 - spm_Ncdf(0,abs(Ep),diag(Cp)) % different from 0
Pp = 1 - spm_Ncdf(0,Ep,diag(Cp)) % greater than 0

% nparams = length(pE);
% Pp = zeros(nparams,1);
% for parameter = 1:nparams
%     % Reduced prior mean
%     rE = pE;
%     rE(parameter) = -32;
% 
%     % Reduced prior covariance
%     rC = pC;
%     rC(parameter,parameter) = 0;
% 
%     % Compare models (reduced - full free energy)
%     dF = spm_log_evidence_reduce(Ep,Cp,pE,pC,rE,rC);
% 
%     % F -> P
%     P = spm_softmax([dF; 0]);
%     Pp(parameter) = P(2);
%     fprintf('Parameter %d: Log bayes factor versus full model: %2.2f, P=%2.2f\n',...
%         parameter,dF,P(2));   
% end

% Save
DCM.Pp = Pp;
DCM_avg = DCM;
save('DCM_avg.mat','DCM_avg');

%% Review model
f1 = review_model2(DCM);
%review_model2(DCM, age, [brain cognition]);
%saveas(f1, 'Parameters.png', 'png')

%% Figure for grant
f2=figure(Position = [0 70 700 1000]);

% Plot cortical brain data
hold on
co = colororder;
for i = 1:nsubjects
    y = Y_brain(:,i);
    k = ~isnan(y);
    x = edges(k);
    y = y(k);
    plot(x,y,'.:','Color',co(1,:),'LineWidth',1/2);

    y = Y_cognition(:,i);
    k = ~isnan(y);
    x = edges(k);
    y = y(k);
    plot(x,y,'.:','Color',co(2,:),'LineWidth',1/2);
end

nans = diag(DCM.Y.Q{1}) < 1;
for p = 1:size(Y.y,2)
    plot(DCM.ages,DCM.yhat(:,p),'LineWidth',3,'Color',co(p,:));
end
xlabel('Age'); ylabel('Z-scored Brain/Cognition')
legend({'Cortical Thickness','Fluid Intelligence'})
%title('Model fit and raw Data');
set(gca,'FontSize',24);
all_y = [brain; cognition];
ylim([min(all_y) max(all_y)])
saveas(f2, 'Fit.png', 'png')

%% PEB
GCM = cell(nsubjects,1);
for i = 1:nsubjects
    DCM = DCM_avg;

    % Set prior for estimate
    DCM.M.pE = DCM_avg.Ep;
    DCM.M.pC = DCM_avg.Cp;

    % Get subject data
    y = [Y_brain(:,i) Y_cognition(:,i)];

    Q = {};
    nv = size(y,2);
    for v = 1:nv
        % Measurement precision (nans have low prior precision)
        q = ones(nbins,1);
        p = isnan(y(:,v));
        q(p) = 1/1024;
        Q{end+1} = diag(q);
        %e = zeros(nv); e(v,v)=1;
        %Q{end+1} = kron(e,diag(q));

        % Set nans to an arbitrary value
        y(p,v) = mean(y(:,v),"omitnan");  
    end
    %Q = blkdiag(Q{1},Q{2});
    DCM.Y.y = y;    
    DCM.Y.Q = Q;    

    DCM.M.nograph = 1;
    DCM.M.noprint = 1;
    % Invert
    %[Ep,Cp,Eh,Ch,F] = variational_laplace(DCM.M,DCM.U,DCM.Y);    
    [Ep,Cp,Eh,Ch,F] = spm_nlsi_GN(DCM.M,DCM.U,DCM.Y);

    % Integrate under posteriors
    [yhat,ty,xhat,tx] = ode_LL(Ep,DCM.M,DCM.U);

    % Pack
    DCM.Ep = Ep;
    DCM.Cp = Cp;
    DCM.Eh = Eh;
    DCM.Ch = Ch;
    DCM.F = F;
    DCM.yhat = yhat;
    DCM.ty   = ty;
    DCM.xhat = xhat;
    DCM.tx   = tx;
    GCM{i} = DCM;
    fprintf('.')
end
fprintf('\n')


% We limit to just state-space model parameters, because the 
% parameter for the observation model has a very wide prior
% which would need its own covariance component in the PEB model.
inc = 1:4;
PEB = spm_dcm_peb(GCM,[],inc);

BMA = spm_dcm_peb_bmc(PEB); % shows P3 not needed

%[RCM,BMR,BMA] = spm_dcm_bmr_all(PEB)

%% Second level analysis

% Random effects average
%Ep  = PEB.Ep;
%Cp  = PEB.Cp;
Ep  = BMA.Ep;
Cp  = BMA.Cp;

% Fixed effects average over subjects for P5 and P6
BPA = spm_dcm_bpa(GCM,true);
for r = setdiff([1:length(BPA.Ep)], inc)
    Ep(r) = BPA.Ep(r);
    Cp(r,r) = BPA.Cp(r,r);
end

% Integrate model under posteriors
[yhat,ty,xhat,tx] = ode_LL(Ep,DCM.M,DCM.U);

% Plot with mean data
DCM = DCM_avg;
DCM.Ep = Ep;
DCM.Cp = Cp;
DCM.yhat = yhat;
DCM.xhat = xhat;

DCM.M.pE = repmat(0.01,1,6); % Just to show tiny bar on plot!

f1 = review_model2(DCM);
set(gca,'FontSize',18)
%legend({'Prior';' ';'Posterior';' '})
text(0.9,1.5,'Prior','color',[0 0 0],'FontSize',24)
text(0.9,1.3,'Posterior','color',[51 153 255]./255,'FontSize',24)
saveas(f1, 'Parameters.png', 'png')