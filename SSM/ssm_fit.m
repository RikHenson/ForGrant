%% Demonstration of fitting a 2-state model to the OASIS dataset
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

%% Decide the number of evenly spaced age bins
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

%% Fit grand mean (to estimate better priors for each subject later)

% Starting values for brain and cognitive states 
x0 = [1 1]';

% Define model M
M.IS = @ode_LL;
M.f  = @fx_model2;
M.g  = @gx_model2;
M.x  = x0; % starting values
M.l  = 2;  % number of outputs
M.m  = 0;  % number of inputs 
M.ns = nbins; % number of samples

% Data structure
U = struct();
U.u = zeros(nbins,M.m);
U.dt = 1;

% Parameters (see fx_model2.m and gx_model2.m):
% 1 = Brain decline (alpha_1)
% 2 = Cognitive decline (alpha_2)
% 3 = Brain->Cognitive (beta_1)
% 4 = Cognitive->Brain (beta_2)
% 5 = Brain intercept (delta_1)
% 6 = Cognitive intercept (delta_2)

% Priors on parameters
M.pE = [0 0 0 0 0 0]';                     % shrinkage priors
M.pC = diag([1/16 1/16 1/16 1/16 1 1]);    % weak priors on intercepts
%M.pE = [3/2 1 1/2 3/2 0 0]'; % chosen to resemble mean data roughly
%M.pC = diag([1/16 1/16 1/16 1/16 64 64]);

% (Log) Priors on noise hyperparameters (related to SNR of data; could optimise with model evidence)
M.hE = 6;
M.hC = 1/6;

% Measurement precision, based on number of observations per bin (with nans having min of 1/1024)
Q = {}; % Q(i,i) = N where N is the number of measurements 
Q{1} = diag(max(1/1024,N_mean_brain));
Q{2} = diag(max(1/1024,N_mean_cognition));

% Set data
Y = struct();
Y.y  = [Y_mean_brain Y_mean_cognition];
Y.y(isnan(Y.y(:,1)),1) = mean(Y.y(:,1),"omitnan");
Y.y(isnan(Y.y(:,2)),2) = mean(Y.y(:,2),"omitnan");
Y.dt = 1;
Y.Q  = Q;

% Invert (estimate)
M.X0 = ones(size(Y.y,1),1); % only confound is mean over time
[Ep,Cp,Eh,Ch,F] = spm_nlsi_GN(M,U,Y);

% Integrate under posteriors
[yhat,ty,xhat,tx] = ode_LL(Ep,M,U);

% Convert to SPM's "DCM" format for PEB and plotting
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

DCM_avg = DCM;
save('DCM_avg.mat','DCM_avg');

%% Review model
f1 = review_model2(DCM, age, [brain cognition]);
%saveas(f1, 'Group_Average_Parameters.png', 'png')

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

%% Now re-estimate every subject using group-average priors
GCM = cell(nsubjects,1);
for i = 1:nsubjects
    DCM = DCM_avg;

    % Set priors from fitting grand-average
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

        % Set nans to an arbitrary value
        y(p,v) = mean(y(:,v),"omitnan");  
    end
    %Q = blkdiag(Q{1},Q{2});
    DCM.Y.y = y;    
    DCM.Y.Q = Q;    

    DCM.M.nograph = 1;
    DCM.M.noprint = 1;
    % Invert
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

%% Run PEB and Bayesian Model Averaging (BMA)
% We limit to just state-space model parameters, because the 
% parameter for the observation model has a very wide prior
% which would need its own covariance component in the PEB model.
inc = 1:4;
PEB = spm_dcm_peb(GCM,[],inc);
BMA = spm_dcm_peb_bmc(PEB); % shows beta_2 not needed

% Random effects average
%Ep  = PEB.Ep;
%Cp  = PEB.Cp;
Ep  = BMA.Ep;
Cp  = BMA.Cp;

% Use fixed effects average over subjects for Parameters 5 and 6
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

f1 = review_model2(DCM);
set(gca,'FontSize',18)
legend({'Prior';'Posterior'})
saveas(f1, 'Parameters.png', 'png')