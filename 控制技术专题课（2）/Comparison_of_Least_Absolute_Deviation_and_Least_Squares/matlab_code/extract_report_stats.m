clear;
clc;

rng(20260424);

projectRoot = fileparts(fileparts(mfilename('fullpath')));
textDir = fullfile(projectRoot, 'output', 'text');
if ~exist(textDir, 'dir')
    mkdir(textDir);
end

outputFile = fullfile(textDir, 'report_stats_raw.txt');
fid = fopen(outputFile, 'w');
if fid < 0
    error('无法创建统计输出文件。');
end
cleanupObj = onCleanup(@() fclose(fid));

noiseLevels = 0.2:0.3:2.0;
outlierFractions = 0:0.05:0.25;
trialCount = 60;

[lnLS, lnL1] = fitCaseLinear(false);
[loLS, loL1] = fitCaseLinear(true);
[pnLS, pnL1] = fitCasePoly(false);
[poLS, poL1] = fitCasePoly(true);

printCase(fid, 'linear_noise', 'LS', lnLS);
printCase(fid, 'linear_noise', 'L1', lnL1);
printCase(fid, 'linear_outlier', 'LS', loLS);
printCase(fid, 'linear_outlier', 'L1', loL1);
printCase(fid, 'poly_noise', 'LS', pnLS);
printCase(fid, 'poly_noise', 'L1', pnL1);
printCase(fid, 'poly_outlier', 'LS', poLS);
printCase(fid, 'poly_outlier', 'L1', poL1);

[ls1, l11] = sweepNoise(linspace(-4, 4, 36)', linspace(-4, 4, 280)', [1.20; 2.35], 1, noiseLevels, trialCount);
[ls2, l12] = sweepOutlier(linspace(-4, 4, 36)', linspace(-4, 4, 280)', [1.20; 2.35], 1, outlierFractions, 0.55, trialCount);
[ls3, l13] = sweepNoise(linspace(-2.6, 2.6, 40)', linspace(-2.8, 2.8, 320)', [0.65; -1.10; 0.90; 1.80], 3, noiseLevels, trialCount);
[ls4, l14] = sweepOutlier(linspace(-2.6, 2.6, 40)', linspace(-2.8, 2.8, 320)', [0.65; -1.10; 0.90; 1.80], 3, outlierFractions, 0.75, trialCount);

printSweep(fid, 'linear_noise', noiseLevels, ls1, l11);
printSweep(fid, 'linear_outlier', outlierFractions, ls2, l12);
printSweep(fid, 'poly_noise', noiseLevels, ls3, l13);
printSweep(fid, 'poly_outlier', outlierFractions, ls4, l14);

fprintf('统计导出完成: %s\n', outputFile);

function printCase(fid, name, method, metrics)
    fprintf(fid, 'CASE|%s|%s|%.6f|%.6f|%.6f|%.6f\n', ...
        name, method, metrics(1), metrics(2), metrics(3), metrics(4));
end

function printSweep(fid, name, x, ls, l1)
    fprintf(fid, 'STAT|%s|LS|%.6f|%.6f|%.6f|%.6f|%d\n', ...
        name, mean(ls), std(ls), min(ls), max(ls), numel(ls));
    fprintf(fid, 'STAT|%s|L1|%.6f|%.6f|%.6f|%.6f|%d\n', ...
        name, mean(l1), std(l1), min(l1), max(l1), numel(l1));

    [~, idxBestLS] = min(ls);
    [~, idxWorstLS] = max(ls);
    [~, idxBestL1] = min(l1);
    [~, idxWorstL1] = max(l1);

    fprintf(fid, 'EXT|%s|LS|%.6f|%.6f|%.6f|%.6f\n', ...
        name, x(idxBestLS), ls(idxBestLS), x(idxWorstLS), ls(idxWorstLS));
    fprintf(fid, 'EXT|%s|L1|%.6f|%.6f|%.6f|%.6f\n', ...
        name, x(idxBestL1), l1(idxBestL1), x(idxWorstL1), l1(idxWorstL1));
end

function [mLS, mL1] = fitCaseLinear(withOutlier)
    x = linspace(-4.5, 4.5, 45)';
    betaTrue = [1.20; 2.35];
    X = [ones(size(x)), x];
    yTrue = X * betaTrue;
    y = yTrue + 0.75 * randn(size(x));
    if withOutlier
        idx = [6, 15, 29, 38];
        y(idx) = y(idx) + [7.5; -6.8; 8.6; -7.2];
    end
    xDense = linspace(min(x), max(x), 300)';
    XDense = [ones(size(xDense)), xDense];
    yDenseTrue = XDense * betaTrue;
    betaLS = X \ y;
    betaL1 = lad(X, y);
    mLS = metrics(X, y, betaLS, XDense, yDenseTrue, betaTrue);
    mL1 = metrics(X, y, betaL1, XDense, yDenseTrue, betaTrue);
end

function [mLS, mL1] = fitCasePoly(withOutlier)
    x = linspace(-2.8, 2.8, 49)';
    betaTrue = [0.65; -1.10; 0.90; 1.80];
    X = polyDesign(x, 3);
    yTrue = X * betaTrue;
    y = yTrue + 0.95 * randn(size(x));
    if withOutlier
        idx = [8, 19, 34, 44];
        y(idx) = y(idx) + [-9.5; 8.2; -7.6; 10.1];
    end
    xDense = linspace(min(x), max(x), 400)';
    XDense = polyDesign(xDense, 3);
    yDenseTrue = XDense * betaTrue;
    betaLS = X \ y;
    betaL1 = lad(X, y);
    mLS = metrics(X, y, betaLS, XDense, yDenseTrue, betaTrue);
    mL1 = metrics(X, y, betaL1, XDense, yDenseTrue, betaTrue);
end

function out = metrics(X, y, beta, XDense, yDenseTrue, betaTrue)
    yHat = X * beta;
    yDense = XDense * beta;
    residual = y - yHat;
    out = [ ...
        norm(beta - betaTrue), ...
        sqrt(mean(residual .^ 2)), ...
        mean(abs(residual)), ...
        sqrt(mean((yDense - yDenseTrue) .^ 2))];
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

function [rmseLS, rmseL1] = sweepNoise(xTrain, xTest, betaTrue, degree, levels, trialCount)
    XTrain = modelDesign(xTrain, degree);
    XTest = modelDesign(xTest, degree);
    yTestTrue = XTest * betaTrue;
    rmseLS = zeros(size(levels));
    rmseL1 = zeros(size(levels));

    for i = 1:numel(levels)
        sigma = levels(i);
        errLS = zeros(trialCount, 1);
        errL1 = zeros(trialCount, 1);
        for k = 1:trialCount
            yTrain = XTrain * betaTrue + sigma * randn(size(xTrain));
            bLS = XTrain \ yTrain;
            bL1 = lad(XTrain, yTrain);
            errLS(k) = sqrt(mean((XTest * bLS - yTestTrue) .^ 2));
            errL1(k) = sqrt(mean((XTest * bL1 - yTestTrue) .^ 2));
        end
        rmseLS(i) = mean(errLS);
        rmseL1(i) = mean(errL1);
    end
end

function [rmseLS, rmseL1] = sweepOutlier(xTrain, xTest, betaTrue, degree, fractions, noiseSigma, trialCount)
    XTrain = modelDesign(xTrain, degree);
    XTest = modelDesign(xTest, degree);
    yTestTrue = XTest * betaTrue;
    sampleCount = numel(xTrain);
    rmseLS = zeros(size(fractions));
    rmseL1 = zeros(size(fractions));

    for i = 1:numel(fractions)
        fraction = fractions(i);
        outlierCount = round(fraction * sampleCount);
        errLS = zeros(trialCount, 1);
        errL1 = zeros(trialCount, 1);
        for k = 1:trialCount
            yTrain = XTrain * betaTrue + noiseSigma * randn(size(xTrain));
            if outlierCount > 0
                idx = randperm(sampleCount, outlierCount);
                spike = (7 + 3 * rand(outlierCount, 1)) .* sign(randn(outlierCount, 1));
                yTrain(idx) = yTrain(idx) + spike;
            end
            bLS = XTrain \ yTrain;
            bL1 = lad(XTrain, yTrain);
            errLS(k) = sqrt(mean((XTest * bLS - yTestTrue) .^ 2));
            errL1(k) = sqrt(mean((XTest * bL1 - yTestTrue) .^ 2));
        end
        rmseLS(i) = mean(errLS);
        rmseL1(i) = mean(errL1);
    end
end
