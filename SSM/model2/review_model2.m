function f = review_model2(DCM, x, y);
% Review fitted model

if nargin<2
    Y = DCM.Y;
    ages = DCM.ages;
else
    Y.y = y;
    ages = x;
end

M = DCM.M;
Ep = DCM.Ep;
Cp = DCM.Cp;

xhat = DCM.xhat;
yhat = DCM.yhat;

% Identify missing observations

% Plot grand average model fit results
% -------------------------------------------------------------------------
%spm_figure('GetWin','Model fit');
%spm_clf;
figure(Position = [0 140 700 1000])

% Plot data and fit
%subplot(2,1,1);
col = colororder;
hold on;
for p = 1:size(Y.y,2)
    nans = isnan(Y.y(:,p));
    plot(ages(~nans),Y.y(~nans,p),'.','Color',col(p,:),'MarkerSize',6);
end

nans = diag(DCM.Y.Q{1}) < 1;
for p = 1:size(Y.y,2)
    plot(DCM.ages,yhat(:,p),'LineWidth',2,'Color',col(p,:));
end
xlabel('Age'); ylabel('Z-scored Brain/Cognition')
legend({'Brain','Cognition'})
title('Data and Model fit');
set(gca,'FontSize',12);
ylim([min(Y.y(:)) max(Y.y(:))])

% Plot latent states
% subplot(3,1,2);
% plot(DCM.ages,xhat*100,'LineWidth',3);
% xlabel('Age');title('Latent variables');
% legend({'Brain','Cognition'});
% set(gca,'FontSize',12);

% Plot parameters
%subplot(2,1,2);
f = figure(Position = [0 140 700 500]);
%plot_parameters([M.pE Ep],[diag(M.pC) diag(Cp)]);
reord = 1:4;
%reord = [1 4 2 3]; % group parameters better
plot_parameters(Ep(reord),full(Cp(reord,reord)),M.pE(reord),M.pC(reord,reord),{'\alpha_1','\alpha_2','\beta_1','\beta_2'}); % ignoring uncertainty on priors for moment
%spm_plot_ci([M.pE Ep],[diag(M.pC) diag(Cp)]);
xlabel('Parameter'); %title('Parameters');
%ylim([-10 10]);
set(gca,'FontSize',12);
