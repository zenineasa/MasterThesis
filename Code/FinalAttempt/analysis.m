%% Comparing initial guess network

disp(' ');
disp(' ');
disp('******************************************************************');
disp('Initial guess')
disp('******************************************************************');
disp(' ');

% Figure related
close all;
figureIndex = 2000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

% Load data
baseline1 = readmatrix('../experiment/baselineToBeatForInitialGuess.csv');
baseline2 = readmatrix('../experiment/simpleStatisticalBaselines.csv');

cost_array_boundary = baseline1(:, 3);
cost_mean_method = baseline2(:, 2);
cost_median_method = baseline2(:, 3);

initGuessNet = readmatrix('csv/initGuessNetwork_test.csv');
cost_initGuessNet = initGuessNet(:, 2);

% Analyze

ipoptCSV = readmatrix('../experiment/ipoptInformation.csv');
foundOptimalSolution = (ipoptCSV(:, end) == 1);

cost_mean_method = cost_mean_method(foundOptimalSolution);
cost_median_method = cost_median_method(foundOptimalSolution);
cost_array_boundary = cost_array_boundary(foundOptimalSolution);
cost_initGuessNet = cost_initGuessNet(foundOptimalSolution);

disp(' ');
disp('------------------------------------------------------------------');
disp('Mean cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('Mean boundary:');
disp(mean(cost_mean_method));
disp('Median boundary:');
disp(mean(cost_median_method));
disp('Array boundary:');
disp(mean(cost_array_boundary));
disp('Our initial guess network:');
disp(mean(cost_initGuessNet));

disp(' ');
disp('------------------------------------------------------------------');
disp('Median cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('Mean boundary:');
disp(median(cost_mean_method));
disp('Median boundary:');
disp(median(cost_median_method));
disp('Array boundary:');
disp(median(cost_array_boundary));
disp('Our initial guess network:');
disp(median(cost_initGuessNet));

disp(' ');
disp('------------------------------------------------------------------');
disp('Lowest cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('Mean boundary:');
disp(min(cost_mean_method));
disp('Median boundary:');
disp(min(cost_median_method));
disp('Array boundary:');
disp(min(cost_array_boundary));
disp('Our initial guess network:');
disp(min(cost_initGuessNet));

disp(' ');
disp('------------------------------------------------------------------');
disp('Highest cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('Mean boundary:');
disp(max(cost_mean_method));
disp('Median boundary:');
disp(max(cost_median_method));
disp('Array boundary:');
disp(max(cost_array_boundary));
disp('Our initial guess network:');
disp(max(cost_initGuessNet));

disp(' ');
disp('------------------------------------------------------------------');
disp('Histograms and cumulative histograms')
disp('------------------------------------------------------------------');

% Untrimmed cumulative histogram
numBins = 100;

[meanCounts, x1] = histcounts( ...
    cost_mean_method, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[medianCounts, x2] = histcounts( ...
    cost_median_method, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[arrayBoundaryCounts, x3] = histcounts( ...
    cost_array_boundary, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[initGuessCounts, x4] = histcounts( ...
    cost_initGuessNet, 'Normalization', 'cumcount', 'NumBins', numBins ...
);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(x1(1:end-1), meanCounts);
plot(x2(1:end-1), medianCounts);
plot(x3(1:end-1), arrayBoundaryCounts);
plot(x4(1:end-1), initGuessCounts);
legend({'Mean', 'Median', 'Edge', 'Network'}, 'Location','southeast')
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'initialGuessCumulativeHistogram_untrimmed.eps'], 'epsc')

% Let's limit it to a cost value; we are interested in which has the
% lower cost anyway.
maxCostOfInterest = 2;

% Trimming

cost_mean_method = cost_mean_method(cost_mean_method < maxCostOfInterest);
cost_median_method = cost_median_method(cost_median_method < maxCostOfInterest);
cost_array_boundary = cost_array_boundary(cost_array_boundary < maxCostOfInterest);
cost_initGuessNet = cost_initGuessNet(cost_initGuessNet < maxCostOfInterest);

% Histogram for each of these

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(cost_mean_method, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'initialGuessHistogram_mean.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(cost_median_method, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'initialGuessHistogram_median.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(cost_array_boundary, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'initialGuessHistogram_edge.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(cost_initGuessNet, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'initialGuessHistogram_network.eps'], 'epsc')

% Cumulative histogram for all of these put together

[meanCounts, x1] = histcounts( ...
    cost_mean_method, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[medianCounts, x2] = histcounts( ...
    cost_median_method, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[arrayBoundaryCounts, x3] = histcounts( ...
    cost_array_boundary, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[initGuessCounts, x4] = histcounts( ...
    cost_initGuessNet, 'Normalization', 'cumcount', 'NumBins', numBins ...
);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(x1(1:end-1), meanCounts);
plot(x2(1:end-1), medianCounts);
plot(x3(1:end-1), arrayBoundaryCounts);
plot(x4(1:end-1), initGuessCounts);
legend({'Mean', 'Median', 'Edge', 'Network'}, 'Location','southeast')
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'initialGuessCumulativeHistogram.eps'], 'epsc')


%% Comparing spatio temporal optimizer

disp(' ');
disp(' ');
disp('******************************************************************');
disp('Optimizer')
disp('******************************************************************');
disp(' ');

% Figure related
close all;
figureIndex = 3000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

ipoptCSV = readmatrix('../experiment/ipoptInformation.csv');
sgdCSV = readmatrix('../experiment/gradientDescendWithoutInput.csv');
adamCSV = readmatrix('../experiment/adamWithoutInput.csv');
optimizerNetworkCSV = readmatrix('csv/optimizerNetwork_test.csv');
extendedCSV = readmatrix('csv/optimizerNetwork_extended_test.csv');

optimNetCost = optimizerNetworkCSV(:, 2);
ipoptCost = ipoptCSV(:, 2);
sgdCost = sgdCSV(:, 3);
adamCost = adamCSV(:, 3);
extendedCost = extendedCSV(:, 2);

% When we compare with IPOPT, only compare with the converged ones.
foundOptimalSolution = (ipoptCSV(:, end) == 1);

% SGD and Adam baselines have only have 400 data points. MaxIndex takes
% care of that.
maxIndex = min([ ...
    length(ipoptCost), length(sgdCost), length(adamCost) ...
    length(optimNetCost), length(extendedCost), ...
]);
foundOptimalSolution = foundOptimalSolution(1:maxIndex);

ipoptCost = ipoptCost(foundOptimalSolution);
sgdCost = sgdCost(foundOptimalSolution);
adamCost = adamCost(foundOptimalSolution);
optimNetCost = optimNetCost(foundOptimalSolution);
extendedCost = extendedCost(foundOptimalSolution);


disp(' ');
disp('------------------------------------------------------------------');
disp('Mean cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('IPOPT:');
disp(mean(ipoptCost));
disp('SGD:');
disp(mean(sgdCost));
disp('Adam:');
disp(mean(adamCost));
disp('Our optimizer network (8 iterations):');
disp(mean(optimNetCost));
disp('Our optimizer network (32 iterations):');
disp(mean(extendedCost));

disp(' ');
disp('------------------------------------------------------------------');
disp('Median cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('IPOPT:');
disp(median(ipoptCost));
disp('SGD:');
disp(median(sgdCost));
disp('Adam:');
disp(median(adamCost));
disp('Our optimizer network (8 iterations):');
disp(median(optimNetCost));
disp('Our optimizer network (32 iterations):');
disp(median(extendedCost));

disp(' ');
disp('------------------------------------------------------------------');
disp('Lowest cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('IPOPT:');
disp(min(ipoptCost));
disp('SGD:');
disp(min(sgdCost));
disp('Adam:');
disp(min(adamCost));
disp('Our optimizer network (8 iterations):');
disp(min(optimNetCost));
disp('Our optimizer network (32 iterations):');
disp(min(extendedCost));

disp(' ');
disp('------------------------------------------------------------------');
disp('Highest cost (for problems that IPOPT found feasible solution)')
disp('------------------------------------------------------------------');

disp('IPOPT:');
disp(max(ipoptCost));
disp('SGD:');
disp(max(sgdCost));
disp('Adam:');
disp(max(adamCost));
disp('Our optimizer network (8 iterations):');
disp(max(optimNetCost));
disp('Our optimizer network (32 iterations):');
disp(max(extendedCost));

disp(' ');
disp('------------------------------------------------------------------');
disp('Histograms and cumulative histograms')
disp('------------------------------------------------------------------');

numBins = 100;
% Let's limit it to a cost value; we are interested in which has the
% lower cost anyway.
maxCostOfInterest = 0.8;

ipoptCost = ipoptCost(ipoptCost < maxCostOfInterest);
sgdCost = sgdCost(sgdCost < maxCostOfInterest);
adamCost = adamCost(adamCost < maxCostOfInterest);
optimNetCost = optimNetCost(optimNetCost < maxCostOfInterest);
extendedCost = extendedCost(extendedCost < maxCostOfInterest);

% Histogram for each of these

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(ipoptCost, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'optimizerNetHistogram_ipopt.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(sgdCost, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'optimizerNetHistogram_sgd.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(adamCost, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'optimizerNetHistogram_adam.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(optimNetCost, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'optimizerNetHistogram_optimNet.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(extendedCost, 'NumBins', numBins)
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'optimizerNetHistogram_extended.eps'], 'epsc')

% Cumulative histogram for all of these put together

[ipoptCounts, x1] = histcounts( ...
    ipoptCost, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[sgdCounts, x2] = histcounts( ...
    sgdCost, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[adamCounts, x3] = histcounts( ...
    adamCost, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[optimNetCounts, x4] = histcounts( ...
    optimNetCost, 'Normalization', 'cumcount', 'NumBins', numBins ...
);
[extendedCounts, x5] = histcounts( ...
    extendedCost, 'Normalization', 'cumcount', 'NumBins', numBins ...
);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(x1(1:end-1), ipoptCounts);
plot(x2(1:end-1), sgdCounts);
plot(x3(1:end-1), adamCounts);
plot(x4(1:end-1), optimNetCounts);
plot(x5(1:end-1), extendedCounts);
legend({'IPOPT', 'SGD', 'Adam', 'Network - 8 iterations', 'Network - 32 iterations'}, ...
    'Location','southeast')
xlabel('Cost')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'optimizerNetCumulativeHistogram.eps'], 'epsc')


%% Extended testing related

disp(' ');
disp(' ');
disp('******************************************************************');
disp('Extended')
disp('******************************************************************');
disp(' ');

figureIndex = 3000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

% i,minNetCost,minNetCostIter,minNetMaxAbsCV,minNetMeanSquaredCV,minCostWithoutCV,minCostWithoutCVIter,beatIPOPTIter,beatIPOPTWithZeroViolationIter

extendedCSV = readmatrix('csv/optimizerNetwork_extended_test.csv');
ipoptCSV = readmatrix('../experiment/ipoptInformation.csv');

ipoptCost = ipoptCSV(:, 2);
extendedCost = extendedCSV(:, 2);
extendedCostWithoutCV = extendedCSV(:, 6);

foundOptimalSolution = (ipoptCSV(:, end) == 1);

minNetMaxAbsCV = extendedCSV(:, 4);
minNetMeanSquaredCV = extendedCSV(:, 5);
minNetMaxAbsCV = minNetMaxAbsCV(foundOptimalSolution);
minNetMeanSquaredCV = minNetMeanSquaredCV(foundOptimalSolution);


beatIPOPTWithCVIter = extendedCSV(:, end-1);
beatIPOPTWithoutCVIter = extendedCSV(:, end);
minCostWithCVIter = extendedCSV(:, 3);
minCostWithoutCVIter = extendedCSV(:, end-2);


disp(' ');
disp('------------------------------------------------------------------');
disp('Simple statistical information')
disp('------------------------------------------------------------------');

disp('Number of cases where method beats IPOPT with some violation allowed:')
disp(nnz(beatIPOPTWithCVIter(foundOptimalSolution) ~= -1))
disp('Number of cases where method beats IPOPT without any violation:')
disp(nnz(beatIPOPTWithoutCVIter(foundOptimalSolution) ~= -1))

disp('Percent cases where method beats IPOPT with some violation allowed:')
disp(nnz(beatIPOPTWithCVIter(foundOptimalSolution) ~= -1) * 100 ...
    / nnz(foundOptimalSolution))
disp('Percent cases where method beats IPOPT without any violation:')
disp(nnz(beatIPOPTWithoutCVIter(foundOptimalSolution) ~= -1) * 100 ...
    / nnz(foundOptimalSolution))

disp(' ');
disp('------------------------------------------------------------------');
disp('Iteration at which ...')
disp('------------------------------------------------------------------');

% Lowest cost at which step
figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(minCostWithCVIter(beatIPOPTWithCVIter ~= -1))
xlabel('Number of iterations')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'lowestCostIterHistogram.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(minCostWithoutCVIter(beatIPOPTWithoutCVIter ~= -1))
xlabel('Number of iterations')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'lowestCostWithoutCVIterHistogram.eps'], 'epsc')

% Steps at which the method beats IPOPT
figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(beatIPOPTWithCVIter(beatIPOPTWithCVIter ~= -1))
xlabel('Number of iterations')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'methodBeatsIPOPTIterHistogram.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
histogram(beatIPOPTWithoutCVIter(beatIPOPTWithoutCVIter ~= -1))
xlabel('Number of iterations')
ylabel('Number of problems')
grid on;
grid minor;
saveas(fig, [savePath 'methodBeatsIPOPTWithoutCVIterHistogram.eps'], 'epsc')

%%
disp(' ');
disp('------------------------------------------------------------------');
disp('Scenarios where edge values worked better than initial guess network')
disp('------------------------------------------------------------------');

figureIndex = 4000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';
desiredProfileInfoFileName = 'desiredProfileInfo.mat';

problems = readmatrix('../data/data.csv', 'OutputType', 'string');

ipoptCSV = readmatrix('../experiment/ipoptInformation.csv');
foundOptimalSolution = (ipoptCSV(:, end) == 1);

baseline1 = readmatrix('../experiment/baselineToBeatForInitialGuess.csv');
cost_array_boundary = baseline1(:, 3);

initGuessNet = readmatrix('csv/initGuessNetwork_test.csv');
cost_initGuessNet = initGuessNet(:, 2);

cost_array_boundary = cost_array_boundary(foundOptimalSolution);
cost_initGuessNet = cost_initGuessNet(foundOptimalSolution);

% Generate the desired profile from the equation
% And extract the values needed for correlative analysis
profileEquations = problems(:, 9);
Ns = problems(:, 2);

if ~isfile(desiredProfileInfoFileName)
    disp('Getting info on desired profile; this will take a while...');
    array_min = zeros(size(Ns));
    array_max = zeros(size(Ns));
    array_mean = zeros(size(Ns));
    array_median = zeros(size(Ns));

    for iter = 1:length(Ns)
        N = str2double(Ns(iter));
        h = 1/(N+1);
        y_desired = zeros(N+2, N+2);
        for i = 0:(N+1)
            for j = 0:(N+1)
                x1 = h * i;
                x2 = h * j;
                y_desired(i+1, j+1) = eval(profileEquations(iter));
            end
        end

        array_min(iter) = min(y_desired, [], 'all');
        array_max(iter) = max(y_desired, [], 'all');
        array_mean(iter) = mean(y_desired, 'all');
        array_median(iter) = median(y_desired, 'all');
    end

    save(desiredProfileInfoFileName, "array_min", "array_max", "array_mean", "array_median");
else
    load(desiredProfileInfoFileName);
end

costDiff = cost_array_boundary - cost_initGuessNet;

array_min = array_min(foundOptimalSolution);
array_max = array_max(foundOptimalSolution);
array_mean = array_mean(foundOptimalSolution);
array_median = array_median(foundOptimalSolution);

bounds = problems(:, 4:7);
bounds = str2double(bounds(foundOptimalSolution, :));

d_const = problems(:, 8);
d_const = str2double(d_const(foundOptimalSolution));

% Calculating partial correlations

x = [array_min array_max array_mean array_median, ...
    bounds, d_const, costDiff];
rho = partialcorr(x);
disp([ ...
    'Partial correlations between min, max, mean, median of desired profile matrix, '...
    'upper and lower bounds of domain and boundary values, sourcing term, and, ' ...
    'cost difference between edge and initial guess network:'
    ])
disp(rho);

disp(['The strongest correlation between costDiff and any other variable' ...
    ' observed is with d_const, which is about -0.4593. There is no ' ...
    'strong correlation with any other variable.']);

% Visualizing how d_const affects the methods
dConstValues = unique(d_const)';
initGuessNetWins = zeros(size(dConstValues));
edgeValuesWins = zeros(size(dConstValues));

for i = 1:length(dConstValues)
    costDiffForSourcingTerm = costDiff(d_const == dConstValues(i));
    disp(['For term: ' num2str(dConstValues(i))])
    initGuessNetWins(i) = nnz(costDiffForSourcingTerm >= 0);
    edgeValuesWins(i) = nnz(costDiffForSourcingTerm <= 0);
end

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
bar(dConstValues, [initGuessNetWins; edgeValuesWins]');
xlabel('Sourcing term values')
ylabel('Number of problems')
legend({'Initial Guess Network has lower cost', 'Edge has lower cost'}, ...
    'Location','northwest')
grid on;
grid minor;
saveas(fig, [savePath 'InitGuessNetworkVSEdge.eps'], 'epsc')


%%
disp(' ');
disp('------------------------------------------------------------------');
disp('Scenarios where Optimizer Network performs poorly than IPOPT')
disp('------------------------------------------------------------------');

figureIndex = 5000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';
desiredProfileInfoFileName = 'desiredProfileInfo.mat';

problems = readmatrix('../data/data.csv', 'OutputType', 'string');

ipoptCSV = readmatrix('../experiment/ipoptInformation.csv');
extendedCSV = readmatrix('csv/optimizerNetwork_extended_test.csv');
load(desiredProfileInfoFileName);

ipoptCost = ipoptCSV(:, 2);
extendedCost = extendedCSV(:, 2);

foundOptimalSolution = (ipoptCSV(:, end) == 1);

ipoptCost = ipoptCost(foundOptimalSolution);
extendedCost = extendedCost(foundOptimalSolution);

costDiff = ipoptCost - extendedCost;

array_min = array_min(foundOptimalSolution);
array_max = array_max(foundOptimalSolution);
array_mean = array_mean(foundOptimalSolution);
array_median = array_median(foundOptimalSolution);

bounds = problems(:, 4:7);
bounds = str2double(bounds(foundOptimalSolution, :));

d_const = problems(:, 8);
d_const = str2double(d_const(foundOptimalSolution));

% Calculating partial correlations

x = [array_min array_max array_mean array_median, ...
    bounds, d_const, costDiff];
rho = partialcorr(x);
disp([ ...
    'Partial correlations between min, max, mean, median of desired profile matrix, '...
    'upper and lower bounds of domain and boundary values, sourcing term, and, ' ...
    'cost difference between IPOPT and Optimizer network:'
    ])
disp(rho);

disp('Looks like there is no correlation strong enough to be analyzed.')

%%
upperLimit = max(ipoptCost);
delta = 0.1 * upperLimit;

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
scatter(ipoptCost, extendedCost, 'o', 'MarkerEdgeAlpha',0.3)

plot([0 upperLimit], [0 upperLimit], '--')
plot([0+delta upperLimit], [0 upperLimit-delta], 'k--')
plot([0-delta upperLimit], [0 upperLimit+delta], 'k--')

xlim([0 upperLimit])
ylim([0 upperLimit])

xlabel('IPOPT Cost');
ylabel('Method Cost');

grid on;
grid minor;
saveas(fig, [savePath 'OptimizerNetworkVSIPOPT.eps'], 'epsc')

% Count the number of points between those two parallel lines.
disp('The total number of points within the specified region is')
disp( ...
    nnz( ...
        (extendedCost <= ipoptCost + delta) & ...
        (extendedCost >= ipoptCost - delta) ...
    ) ...
)

%%

figureIndex = 5500;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

ipoptCSV = readmatrix('../experiment/ipoptInformation.csv');
extendedCSV = readmatrix('csv/optimizerNetwork_extended_test.csv');

%errors = ipoptCSV(:, [7, 9, 11, 13]);
ipoptConstraintViolation = ipoptCSV(:, 9);
methodMinNetMeanSquaredCV = extendedCSV(:, 5);
ipoptCost = ipoptCSV(:, 2);
extendedCost = extendedCSV(:, 2);

foundOptimalSolution = (ipoptCSV(:, end) == 1);

ipoptConstraintViolation = ipoptConstraintViolation(foundOptimalSolution);
methodMinNetMeanSquaredCV = methodMinNetMeanSquaredCV(foundOptimalSolution);
ipoptCost = ipoptCost(foundOptimalSolution);
extendedCost = extendedCost(foundOptimalSolution);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
scatter(ipoptConstraintViolation, methodMinNetMeanSquaredCV, 'o', 'MarkerEdgeAlpha',0.3)
xlabel('IPOPT Violation');
ylabel('Method Violation');
grid on;
grid minor;
saveas(fig, [savePath 'OptimizerNetworkVSIPOPTViolation.eps'], 'epsc')

%upperLimit = min(max(ipoptOverallNLPError), max(methodMinNetMeanSquaredCV));
%xlim([0 upperLimit])
%ylim([0 upperLimit])
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
grid on;
grid minor;
saveas(fig, [savePath 'OptimizerNetworkVSIPOPTViolationLOGLOG.eps'], 'epsc')

% Cost difference vs violation difference

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
scatter(ipoptCost-extendedCost, ipoptConstraintViolation-methodMinNetMeanSquaredCV, 'o', 'MarkerEdgeAlpha',0.5)
xlabel('Cost difference (IPOPT cost - Method cost)');
ylabel('Violation difference (IPOPT violation - Method violation)');
grid on;
grid minor;
saveas(fig, [savePath 'OptimizerNetworkVSIPOPTCostsDiffVsViolationDiff.eps'], 'epsc')


%%

disp(' ');
disp('------------------------------------------------------------------');
disp('Contribution of Adam, RMSProp and Spatio-temporal parts')
disp('------------------------------------------------------------------');

figureIndex = 6000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

% iter,N,adam,rms,net
contribCSV = readmatrix('csv/optimizerNetwork_Contributions.csv');

rho = partialcorr(contribCSV);
disp(rho);

iters = contribCSV(:, 1);
Ns = contribCSV(:, 2);
contribAdam = contribCSV(:, 3);
contribRMSProp = contribCSV(:, 4);
contribNet = contribCSV(:, 5);

% Plot contributions for different Ns
uniqueNs = unique(Ns)';
tempAdamContrib = zeros(size(uniqueNs));
tempRMSPropContrib = zeros(size(uniqueNs));
tempNetContrib = zeros(size(uniqueNs));
for i = 1:length(uniqueNs)
    N = uniqueNs(i);
    tempAdamContrib(i) = mean(contribAdam(N == Ns));
    tempRMSPropContrib(i) = mean(contribRMSProp(N == Ns));
    tempNetContrib(i) = mean(contribNet(N == Ns));
end

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(uniqueNs, tempAdamContrib)
plot(uniqueNs, tempRMSPropContrib)
plot(uniqueNs, tempNetContrib)
xlabel('Domain size');
ylabel('Contribution');
% NOTE: Not saving this; this does not really make much sense to have.

% Plot contributions for different iters
uniqueIters = unique(iters)';
tempAdamContrib = zeros(size(uniqueIters));
tempRMSPropContrib = zeros(size(uniqueIters));
tempNetContrib = zeros(size(uniqueIters));
for i = 1:length(uniqueIters)
    iter = uniqueIters(i);
    tempAdamContrib(i) = mean(contribAdam(iter == iters));
    tempRMSPropContrib(i) = mean(contribRMSProp(iter == iters));
    tempNetContrib(i) = mean(contribNet(iter == iters));
end

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(uniqueIters, tempAdamContrib)
plot(uniqueIters, tempRMSPropContrib)
plot(uniqueIters, tempNetContrib)
xlabel('Iteration');
ylabel('Contribution');
legend({'Adam', 'RMSProp', 'Spatio-temporal'}, ...
    'Location','northeast')
grid on;
grid minor;
saveas(fig, [savePath 'contributionByAdamRMSPropAndNet.eps'], 'epsc')

netDeltas = contribCSV(:, end);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on
plot(netDeltas(1:32))
plot(netDeltas(33:64))
plot(netDeltas(65:96))
plot(netDeltas(97:128))
xlabel('Iteration');
ylabel('Contribution by Spatio-temporal part');
grid on;
grid minor;
saveas(fig, [savePath 'contributionByNetFor4DifferentProblems.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
plot(netDeltas)
xlabel('Iteration');
ylabel('Contribution by Spatio-temporal part');
grid on;
grid minor;
saveas(fig, [savePath 'contributionByNetAllIterations.eps'], 'epsc')

%%
disp(' ');
disp('------------------------------------------------------------------');
disp('Performance')
disp('------------------------------------------------------------------');

figureIndex = 7000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

ipoptPerfCSV = readmatrix('../experiment/ipoptFLOPs.csv');
networkPerfCSV = readmatrix('csv/optimizerNetwork_TIME.csv');

NsIPOPT = ipoptPerfCSV(:, 1);
assert(all(NsIPOPT == networkPerfCSV(:, 1)))

% Timing

timeInitGuess = networkPerfCSV(:, 4);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
plot(NsIPOPT, timeInitGuess, '-o');
set(gca, 'YScale', 'log');
xlabel('Domain size');
ylabel('Time (seconds)');
grid on;
grid minor;
saveas(fig, [savePath 'InitGuessTiming.eps'], 'epsc')

timeOptimizerMethod = networkPerfCSV(:, 7);

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
plot(NsIPOPT, timeOptimizerMethod, '-o');
set(gca, 'YScale', 'log');
xlabel('Domain size');
ylabel('Time (seconds)');
grid on;
grid minor;
saveas(fig, [savePath 'OptimizerTiming.eps'], 'epsc')

timeIPOPT = ipoptPerfCSV(:, 2);
timeNetwork = timeInitGuess + timeOptimizerMethod;

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(NsIPOPT, timeIPOPT, '-o');
plot(NsIPOPT, timeNetwork, '-o');
plot(NsIPOPT, timeNetwork .* 0.1, '--');
plot(NsIPOPT, timeNetwork .* 0.01, '--');
set(gca, 'YScale', 'log');
xlabel('Domain size');
ylabel('Time (seconds)');
legend({'IPOPT', 'Method', 'Method \times 0.1', 'Method \times 0.01'}, ...
    'Location', 'northwest')
grid on;
grid minor;
saveas(fig, [savePath 'MethodVsIPOPTTiming.eps'], 'epsc')

% FLOPs

NsNetworkFLOPs = 10:10:100;

% Initial guess network
addFlops = [100 400 900 1600 2500 3600 4900 6400 8100 10000];
mulFlops = [100 400 900 1600 2500 3600 4900 6400 8100 10000];

initGuessTotalFlops = addFlops + mulFlops;

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;
plot(initGuessTotalFlops, '-o');
xlabel('Domain size');
ylabel('Total FLOPs');
grid on;
grid minor;
saveas(fig, [savePath 'InitGuessFLOPsVsDomainSize.eps'], 'epsc')

% Optimizer network
oneStepAddFlops = [242920 2415568 10311320 28927168 64765192 127939960 224295640 366172016 487729200 602125240];
oneStepMulFlops = [125169 1229105 5229393 14648829 32774771 64708153 113399841 185096589 246513957 304298509];
oneStepConvFlops = [43200 86400 129600 172800 216000 259200 302400 345600 388800 432000];

twoStepAddFlops = [364496 3623568 15469344 43396460 97153712 191914400 336446768 549285736 731611644 903240896];
twoStepMulFlops = [208482 2046466 8702722 24379526 54556098 107695762 188751064 308088818 410402478 506494822];
twoStepConvFlops = [86400 172800 259200 345600 432000 518400 604800 691200 777600 864000];

threeStepAddFlops = [486072 4832052 20623272 57855168 129542232 255888840 448608264 732379000 975528000 1204273000];
threeStepMulFlops = [291507 2863027 12178159 34113751 76321251 150698687 264096963 431054000 574088000 708681000];
threeStepConvFlops = [129600 259200 388800 518400 648000 777600 907200 1037000 1166000 1296000];

fourStepAddFlops = [607648 6040052 25782320 72319168 161914528 319856000 560749000 915459000 1219419000 1505399000];
fourStepMulFlops = [374676 3678576 15648356 43839036 98081046 193675000 339469000 554053000 737892000 910961000];
fourStepConvFlops = [172800 345600 518400 691200 864000 1037000 1210000 1382000 1555000 1728000];

% We can observe that
% twoStepAddFlops - oneStepAddFlops is about equal to threeStepAddFlops - twoStepAddFlops
% This is the number of FLOPS that get added in every iteration.
% The same goes for other two.

% Let's just plot the total FLOPs for 1 to 4 and then for 32
totalOneStepFLOPs = oneStepAddFlops + oneStepMulFlops + oneStepConvFlops;
totalTwoStepFLOPs = twoStepAddFlops + twoStepMulFlops + twoStepConvFlops;
totalThreeStepFLOPs = threeStepAddFlops + threeStepMulFlops + threeStepConvFlops;
totalFourStepFLOPs = fourStepAddFlops + fourStepMulFlops + fourStepConvFlops;

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
plot(NsNetworkFLOPs, ...
    [totalOneStepFLOPs; totalTwoStepFLOPs; totalThreeStepFLOPs; totalFourStepFLOPs]', ...
    '-o');
xlabel('Domain size');
ylabel('Total FLOPs');
legend({'1 Step', '2 Steps', '3 Steps', '4 Steps'}, 'Location','northwest')
grid on;
grid minor;
saveas(fig, [savePath 'OptimizerFLOPsVsDomainSize.eps'], 'epsc')

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
plot([totalOneStepFLOPs; totalTwoStepFLOPs; totalThreeStepFLOPs; totalFourStepFLOPs], '-o');
xlabel('Number of iterations');
ylabel('Total FLOPs');
legend({'N = 10', 'N = 20', 'N = 30', 'N = 40', 'N = 50', 'N = 60', ...
    'N = 70', 'N = 80', 'N = 90', 'N = 100'}, ...
    'Location','northwest')
grid on;
grid minor;
saveas(fig, [savePath 'OptimizerFLOPsVsNumIterations.eps'], 'epsc')

% Comparing both for 32 iterations
ipoptFLOPs = sum(ipoptPerfCSV(:, 4:end), 2);
networkFLOPs = (totalTwoStepFLOPs - totalOneStepFLOPs) * (32 - 1) + ...
    totalOneStepFLOPs + initGuessTotalFlops;

figureIndex = figureIndex + 1;
fig = figure(figureIndex);
hold on;

plot(NsIPOPT, ipoptFLOPs, '-o');
plot(NsNetworkFLOPs, networkFLOPs, '-o');

set(gca, 'YScale', 'log');
xlabel('Domain size');
ylabel('Total FLOPs');

legend({'IPOPT', 'Method'}, 'Location', 'northwest')
grid on;
grid minor;
saveas(fig, [savePath 'MethodVsIPOPTFlops.eps'], 'epsc')

%%
disp(' ');
disp('------------------------------------------------------------------');
disp('Large domain sizes')
disp('------------------------------------------------------------------');

figureIndex = 9000;
savePath = '../../Report/actualContent/figures/ExperimentsResultsAndAnalysis/MATLAB/';

largenCSV = readmatrix('csv/optimizerNetwork_LARGEN.csv');
problemIndices = largenCSV(:, 1);
Ns = largenCSV(:, 2);
costMethod = largenCSV(:, 3);
costIPOPT = largenCSV(:, 4);

uniqueProblemIndices = unique(problemIndices)';
for i = 1:length(uniqueProblemIndices)
    index = uniqueProblemIndices(i);

    figureIndex = figureIndex + 1;
    fig = figure(figureIndex);
    hold on;

    plot(Ns(problemIndices == index), costMethod(problemIndices == index), '-o');
    plot(Ns(problemIndices == index), costIPOPT(problemIndices == index), '-o');

    set(gca, 'YScale', 'log');
    xlabel('Domain size');
    ylabel('Cost');
    legend({'Method', 'IPOPT'}, 'Location','northwest')

    grid on;
    grid minor;
    saveas(fig, [savePath 'ComparingCostForLargeN_' num2str(index) '.eps'], 'epsc')
end
