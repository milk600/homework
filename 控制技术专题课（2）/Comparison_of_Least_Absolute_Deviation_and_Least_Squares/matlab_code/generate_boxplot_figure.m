clear;
clc;
close all;

rng(20260424);

projectRoot = fileparts(fileparts(mfilename('fullpath')));
pictureDir = fullfile(projectRoot, 'output', 'picture');
textDir = fullfile(projectRoot, 'output', 'text');
if ~exist(pictureDir, 'dir')
    mkdir(pictureDir);
end
if ~exist(textDir, 'dir')
    mkdir(textDir);
end

fontName = chooseCJKFont();
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', fontName);
set(groot, 'defaultTextFontName', fontName);
set(groot, 'defaultLegendFontName', fontName);
set(groot, 'defaultAxesFontSize', 11);

trialCount = 120;

[linearNoiseLS, linearNoiseL1] = collectNoiseDistribution(linspace(-4, 4, 36)', linspace(-4, 4, 280)', [1.20; 2.35], 1, 2.0, trialCount);
[linearOutlierLS, linearOutlierL1] = collectOutlierDistribution(linspace(-4, 4, 36)', linspace(-4, 4, 280)', [1.20; 2.35], 1, 0.25, 0.55, trialCount);
[polyNoiseLS, polyNoiseL1] = collectNoiseDistribution(linspace(-2.6, 2.6, 40)', linspace(-2.8, 2.8, 320)', [0.65; -1.10; 0.90; 1.80], 3, 2.0, trialCount);
[polyOutlierLS, polyOutlierL1] = collectOutlierDistribution(linspace(-2.6, 2.6, 40)', linspace(-2.8, 2.8, 320)', [0.65; -1.10; 0.90; 1.80], 3, 0.25, 0.75, trialCount);

allData = [linearNoiseLS; linearNoiseL1; linearOutlierLS; linearOutlierL1; ...
    polyNoiseLS; polyNoiseL1; polyOutlierLS; polyOutlierL1];
group = [ ...
    repmat({'线性-强噪声-L2'}, trialCount, 1); ...
    repmat({'线性-强噪声-L1'}, trialCount, 1); ...
    repmat({'线性-高异常-L2'}, trialCount, 1); ...
    repmat({'线性-高异常-L1'}, trialCount, 1); ...
    repmat({'多项式-强噪声-L2'}, trialCount, 1); ...
    repmat({'多项式-强噪声-L1'}, trialCount, 1); ...
    repmat({'多项式-高异常-L2'}, trialCount, 1); ...
    repmat({'多项式-高异常-L1'}, trialCount, 1)];

fig = figure('Position', [80, 80, 1400, 680]);
boxchart(categorical(group), allData, 'BoxFaceColor', [0.25 0.55 0.85], 'MarkerStyle', '.');
title('L1/L2 回归在高强度干扰下的测试 RMSE 分布比较', 'FontWeight', 'bold');
xlabel('场景与方法');
ylabel('测试 RMSE');
grid on;
box on;
ax = gca;
ax.XTickLabelRotation = 20;
exportgraphics(fig, fullfile(pictureDir, 'rmse_distribution_boxplot_l1_vs_l2.png'), 'Resolution', 300);

summaryFile = fullfile(textDir, 'boxplot_stats_summary.txt');
fid = fopen(summaryFile, 'w');
if fid < 0
    error('无法创建箱型图统计文件。');
end
cleanupObj = onCleanup(@() fclose(fid));
writeSummary(fid, 'linear_noise_L2', linearNoiseLS);
writeSummary(fid, 'linear_noise_L1', linearNoiseL1);
writeSummary(fid, 'linear_outlier_L2', linearOutlierLS);
writeSummary(fid, 'linear_outlier_L1', linearOutlierL1);
writeSummary(fid, 'poly_noise_L2', polyNoiseLS);
writeSummary(fid, 'poly_noise_L1', polyNoiseL1);
writeSummary(fid, 'poly_outlier_L2', polyOutlierLS);
writeSummary(fid, 'poly_outlier_L1', polyOutlierL1);

fprintf('箱型图导出完成: %s\n', fullfile(pictureDir, 'rmse_distribution_boxplot_l1_vs_l2.png'));
fprintf('统计摘要文件: %s\n', summaryFile);

function writeSummary(fid, name, x)
    q = quantile(x, [0.25, 0.50, 0.75]);
    fprintf(fid, '%s|mean=%.6f|std=%.6f|min=%.6f|q1=%.6f|median=%.6f|q3=%.6f|max=%.6f|n=%d\n', ...
        name, mean(x), std(x), min(x), q(1), q(2), q(3), max(x), numel(x));
end

function [errLS, errL1] = collectNoiseDistribution(xTrain, xTest, betaTrue, degree, sigma, trialCount)
    XTrain = modelDesign(xTrain, degree);
    XTest = modelDesign(xTest, degree);
    yTestTrue = XTest * betaTrue;
    errLS = zeros(trialCount, 1);
    errL1 = zeros(trialCount, 1);
    for k = 1:trialCount
        yTrain = XTrain * betaTrue + sigma * randn(size(xTrain));
        betaLS = XTrain \ yTrain;
        betaL1 = lad(XTrain, yTrain);
        errLS(k) = sqrt(mean((XTest * betaLS - yTestTrue) .^ 2));
        errL1(k) = sqrt(mean((XTest * betaL1 - yTestTrue) .^ 2));
    end
end

function [errLS, errL1] = collectOutlierDistribution(xTrain, xTest, betaTrue, degree, fraction, noiseSigma, trialCount)
    XTrain = modelDesign(xTrain, degree);
    XTest = modelDesign(xTest, degree);
    yTestTrue = XTest * betaTrue;
    sampleCount = numel(xTrain);
    outlierCount = round(fraction * sampleCount);
    errLS = zeros(trialCount, 1);
    errL1 = zeros(trialCount, 1);
    for k = 1:trialCount
        yTrain = XTrain * betaTrue + noiseSigma * randn(size(xTrain));
        idx = randperm(sampleCount, outlierCount);
        spike = (7 + 3 * rand(outlierCount, 1)) .* sign(randn(outlierCount, 1));
        yTrain(idx) = yTrain(idx) + spike;
        betaLS = XTrain \ yTrain;
        betaL1 = lad(XTrain, yTrain);
        errLS(k) = sqrt(mean((XTest * betaLS - yTestTrue) .^ 2));
        errL1(k) = sqrt(mean((XTest * betaL1 - yTestTrue) .^ 2));
    end
end

function beta = lad(X, y)
    [n, p] = size(X);
    f = [zeros(p, 1); ones(n, 1)];
    A = [X, -eye(n); -X, -eye(n)];
    b = [y; -y];
    lb = [-inf(p, 1); zeros(n, 1)];
    options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'none');
    z = linprog(f, A, b, [], [], lb, [], options);
    beta = z(1:p);
end

function X = polyDesign(x, degree)
    X = zeros(numel(x), degree + 1);
    for k = 0:degree
        X(:, degree + 1 - k) = x .^ k;
    end
end

function X = modelDesign(x, degree)
    if degree == 1
        X = [ones(size(x)), x];
    else
        X = polyDesign(x, degree);
    end
end

function fontName = chooseCJKFont()
    candidates = {'Microsoft YaHei UI', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS'};
    installed = listfonts;
    fontName = 'Helvetica';
    for i = 1:numel(candidates)
        if any(strcmpi(installed, candidates{i}))
            fontName = candidates{i};
            return;
        end
    end
end
