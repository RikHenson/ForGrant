function h=plot_parameters(Ep,Cp,pE,pC,xlabels)
% Plots the estimated parameters of a model, optionally in a grouped
% bar plot with the parameters used to generate the data.
%
% Ep      - vector of expected values of estimated parameters
% Cp      - covariance matrix of estimated parameters
% pE      - (optional) prior expectation
% pC      - (optional) prior covariance
% xlabels - (optional) labels for the x-axis
%
% E.g.
% figure;
% plot_parameters(P,Cp,pE);
%
% Zeidman, Friston, Parr
% _____________________________________________________________________

h = []; % handles to return

% Validate inputs
P = Ep(:);
if isvector(Cp), Cp = diag(Cp); end
if isvector(pC), pC = diag(pC); end
n  = length(P);

% Default legend text and labels
if nargin < 5
    xlabels = 1:n;
end

% Concatenate generative parameters if provided
has_pE = nargin > 2 && ~isempty(pE);
if has_pE
    pE = pE(:);
end

% Compute 90% confidence interval
ci = spm_invNcdf(1 - 0.05);               % confidence interval
cp  = ci*sqrt(diag(Cp));
cp  = cp(:)';
pc  = ci*sqrt(diag(pC));
pc  = pc(:)';

col = colororder;
%col   = [1 3/4 3/4];

% Plot generative parameters if provided
w       = 0.1;
xoffset = w;
gap     = 0.2;
totalw  = w + gap; % Total width per bar
x = xoffset : totalw : xoffset+((n-1)*totalw);
if has_pE
    h{1} = plot_bars(x,pE,w,col(1,:));
    hold on;
end

% Plot estimated parameters
if has_pE
    xp = x + w;
else
    xp = x;
end
h{2} = plot_bars(xp,P,w,col(2,:));

% Restore zero line
xlims = [0 xp(end)+w+xoffset];
line([xlims(1) xlims(2)],[0 0],'Color','k');

% Add error bars
errx  = xp + (w/2);
perrx = x + (w/2);
for k = 1:n
    h{2} = line([errx(k) errx(k)],[-1 1]*cp(k) + P(k),...
        'LineWidth',5,'Color',col(2,:));
    if has_pE
        h{1} = line([perrx(k) perrx(k)],[-1 1]*pc(k) + pE(k),...
            'LineWidth',2,'Color',col(1,:));
    end
end

% Xlabel
set(gca,'XTick',xp,'XTickLabel',xlabels,'XLim',xlims);

% Ylimits
alldata = [pE; Ep-cp(:); Ep+cp(:); pE-pc(:); pE+pc(:)];
grandmin = min(alldata);
grandmax = max(alldata);
ylim([grandmin-0.05*abs(grandmin) grandmax+0.05*abs(grandmax)]);

hold off;

    function H = plot_bars(x,y,w,c)
        H = [];
        for i = 1:length(x)
            if y(i) == 0
                % Do nothing
            elseif y(i) > 0
                % Positive bar
                H(i) = rectangle('Position',[x(i),0,w,y(i)],'FaceColor',[c 0.5]);
            else
                % Negative bar
                H(i) = rectangle('Position',[x(i),y(i),w,abs(y(i))],'FaceColor',[c 0.5]);
            end
            hold on
        end
    end
end