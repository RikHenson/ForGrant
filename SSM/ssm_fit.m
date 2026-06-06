%% Demonstration of fitting a 2-population model to the OASIS dataset
%
% Rik Henson and Peter Zeidman

%% Import and prepare data
addpath('toolbox');

% whether to plot additional figures
pflag = 0;

% Load data and unpack
load('gm_g.mat');

age = dat.age;
id  = dat.id;

% Get unique subject names
uids = unique(id);
nsubjects = length(uids);

% Zscore data
zscore_xnan = @(x) bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'omitnan')), std(x, 'omitnan'));

brain = zscore_xnan(dat.Cortical_Thickness);
cognition = zscore_xnan(dat.Fluid_Intelligence); % already Z-scored?

% remove couple of outliers
brain(brain < -5) = NaN;

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

%% Visualise data (with spaghetti plot)
if pflag
    figure(Position = [0 70 700 1000])

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
end

%% Fit model to weighted mean over subjects

% Starting values for states 
x0 = [1 1]';

% Model spec
M.IS = @ode_LL;
M.f  = @fx_model2;
M.g  = @gx_model2;
M.x  = x0; % starting values
M.l  = 2;  % number of outputs
M.m  = 1;  % number of inputs 
M.ns = nbins; % number of samples

% Priors on parameters:
% alpha_1 (B decline), beta_1 (B->C), beta_2 (C<-B), alpha_2 (C decline), r_1, r_2 (intercepts))
M.pE = [1 1 1 1 0 0]';
M.pC = diag([1 1 1 1 3 3]);

% Noise priors
M.hE = 6;
M.hC = 1/6;

% Measurement precision (make nans have low prior precision)
Q={};
Q{1} = diag(max(1/1024,N_mean_brain));
Q{2} = diag(max(1/1024,N_mean_cognition));

% Set data
Y = struct();
Y.y  = [Y_mean_brain Y_mean_cognition];
Y.y(isnan(Y.y(:,1)),1) = mean(Y.y(:,1),"omitnan");
Y.y(isnan(Y.y(:,2)),2) = mean(Y.y(:,2),"omitnan");
Y.dt = 1;
Y.Q  = Q;

% Inputs (none here)
U = struct();
U.u = zeros(nbins,M.m);
U.dt = 1;

% Invert
M.X0 = ones(size(Y.y,1),1);
[Ep,Cp,Eh,Ch,F] = spm_nlsi_GN(M,U,Y);

% Plot parameters
reord = [1 4 2 3]; % group parameters better
f1 = plot_parameters(Ep(reord),full(Cp(reord,reord)),M.pE(reord), ...
    {'\alpha_1','\alpha_2','\beta_1','\beta_2'}); 
% ignoring uncertainty on priors for moment
set(gca,'FontSize',24)

saveas(f1, 'Parameters.png', 'png')

%% Plot data and fit 
% (note data just scatter plot, whereas model actually fit to weighted mean
% over subjects)

f2 = figure(Position = [0 70 700 1000]);

% Integrate under posteriors
[yhat,ty,xhat,tx] = ode_LL(Ep,M,U);

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

nans = diag(Y.Q{1}) < 1;
for p = 1:size(Y.y,2)
    plot(edges,yhat(:,p),'LineWidth',3,'Color',co(p,:));
end
xlabel('Age'); ylabel('Z-scored Brain/Cognition')
legend({'Cortical Thickness','Fluid Intelligence'})
%title('Model fit and raw Data');
set(gca,'FontSize',12);
all_y = [brain; cognition];
ylim([min(all_y) max(all_y)])
saveas(f2, 'Fit.png', 'png')