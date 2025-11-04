
clear all; close all; clc;

%% 1. 物理常数和参数
k_B = 8.617333e-5;   % 玻尔兹曼常数 (eV/K)
q = 1.602176e-19;    % 电子电荷 (C)

%% 2. 读取实验数据

% 读取Excel文件
data = xlsread('TSDC_Current.xlsx');
T_exp = data(:,1);     % 温度 (K)
I_exp = abs(data(:,2)); % 电流 (A)，取绝对值避免负值

% 去除零值和异常值
valid_idx = I_exp > 1e-15;  % 去除过小的值
T_exp = T_exp(valid_idx);
I_exp = I_exp(valid_idx);

% 数据平滑处理（使用移动平均）
window_size = 7;  % 增大平滑窗口
I_smooth = smooth(I_exp, window_size, 'moving');

fprintf('数据点数: %d\n', length(T_exp));
fprintf('温度范围: %.2f - %.2f K\n', min(T_exp), max(T_exp));
fprintf('电流范围: %.2e - %.2e A\n', min(I_exp), max(I_exp));

% 估算升温速率（K/s）

beta=0.05; %升温速率3K/min,0.05K/s

%% 3. 单陷阱TSDC电流模型（修正版）
% 基于用户提供的模型形式，正确实现TSDC物理模型
% I(T) = A * exp(-Et/(k*T)) * exp(-nu/beta * integral[T0,T](exp(-Et/(k*T'))dT'))

function I = single_trap_TSDC(T, Et, nu, T0, beta, k_B)
% T: 温度数组 (K)
% Et: 陷阱能级 (eV)
% nu: 频率因子 (1/s)
% T0: 初始温度 (K)
% beta: 升温速率 (K/s)
% k_B: 玻尔兹曼常数 (eV/K)

I = zeros(size(T));

for i = 1:length(T)
    if T(i) > T0
        % 计算积分项
        % 注意：这里的积分是温度的积分，需要转换为时间积分
        integrand = @(T_prime) exp(-Et ./ (k_B * T_prime)) / beta;

        % 数值积分
        integral_val = integral(integrand, T0, T(i), 'RelTol', 1e-6, 'AbsTol', 1e-10);

        % 计算电流
        I(i) = exp(-Et / (k_B * T(i))) * exp(-nu * integral_val);
    else
        I(i) = 0;
    end
end

% 归一化到最大值
if max(I) > 0
    I = I / max(I);
end
end

%% 4. 构建陷阱能级基函数库（修正版）
fprintf('\n构建基函数库...\n');
%E-T关系约为E=0.00311*T-0.02442,在10^13频率因子下
% 调整陷阱能级范围，符合聚合物绝缘材料特性
Et_min = 0.00311*min(T_exp)-0.02442;   % 最小陷阱能级 (eV)
Et_max = 0.00311*max(T_exp)-0.02442;   % 最大陷阱能级 (eV)
dEt = 0.002;    % 能级步长 (eV) - 更精细的步长
Et_array = Et_min:dEt:Et_max;
n_traps = length(Et_array);

fprintf('能级范围: %.2f - %.2f eV\n', Et_min, Et_max);
fprintf('能级步长: %.3f eV\n', dEt);
fprintf('能级点数: %d\n', n_traps);

% 设置固定的初始温度
T0 = 100;  % 初始温度设为100K


nu_j = 10^13; %频率因子
% 构建响应矩阵 A
A = zeros(length(T_exp), n_traps);

fprintf('\n计算各陷阱能级的响应函数...\n');
for j = 1:n_traps
    % 计算该陷阱能级的标准化响应
    A(:,j) = single_trap_TSDC(T_exp, Et_array(j), nu_j, T0, beta, k_B);
end

%% 5. NNLS反卷积求解
fprintf('\n执行NNLS反卷积...\n');

% 归一化实验数据
I_norm = I_smooth / max(I_smooth);

% 设置NNLS选项
options = optimset('Display', 'off', 'TolFun', 1e-12, 'TolX', 1e-12);

% 非负最小二乘法求解
[N_trap, residual_norm] = lsqnonneg(A, I_norm, options);

% 归一化陷阱密度
if max(N_trap) > 0
    N_trap_normalized = N_trap / max(N_trap);
else
    N_trap_normalized = N_trap;
end

% 计算重构电流
I_reconstructed = A * N_trap;

% 计算拟合优度 R²
SS_res = sum((I_norm - I_reconstructed).^2);
SS_tot = sum((I_norm - mean(I_norm)).^2);
R2 = 1 - SS_res / SS_tot;

fprintf('残差范数: %.4e\n', residual_norm);
fprintf('拟合优度 R²: %.4f\n', R2);





%% 6. 结果可视化

figure('Position', [50, 50, 1000, 800], 'Color', 'white');

% 子图1：原始数据和平滑数据
subplot(2,2,1);
semilogy(T_exp, I_exp, 'b.', 'MarkerSize', 2, 'DisplayName', '原始数据');
hold on;
semilogy(T_exp, I_smooth, 'r-', 'LineWidth', 2, 'DisplayName', '平滑数据');
xlabel('温度 (K)', 'FontSize', 11);
ylabel('电流 (A)', 'FontSize', 11);
title('TSDC原始数据', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
xlim([min(T_exp), max(T_exp)]);

% 子图2：陷阱能级-密度分布
subplot(2,2,2);
plot(Et_array, N_trap_normalized, 'b-', 'LineWidth', 2);
hold on;

xlabel('陷阱能级 Et (eV)', 'FontSize', 11);
ylabel('相对陷阱密度', 'FontSize', 11);
title('陷阱能级分布 (NNLS)', 'FontSize', 12, 'FontWeight', 'bold');

grid on;
xlim([Et_min, Et_max]);
ylim([0, 1.1]);

% 子图3：重构验证
subplot(2,2,3);
plot(T_exp, I_norm, 'k-', 'LineWidth', 2, 'DisplayName', '实验数据');
hold on;
plot(T_exp, I_reconstructed, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('重构 (R²=%.3f)', R2));
xlabel('温度 (K)', 'FontSize', 11);
ylabel('归一化电流', 'FontSize', 11);
title('重构验证', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
xlim([min(T_exp), max(T_exp)]);

%% 7.陷阱密度计算

%陷阱密度,单位（m^-3）
N_density=trapz(T_exp,I_exp)*(1/beta)*(6.24e18)*(1/3.14e-8);

%Gaussian基准函数

x=-0.2:dEt:0.2;
sigma=0.02;
Gauss_base=1/(sqrt(2*pi)*sigma)*exp(-(x).^2/(2*sigma^2));

%对冲击函数进行展宽 用conv卷积
N_trap_widening=conv(N_trap,Gauss_base,"same").';
k_density=N_density/(trapz(Et_array,N_trap_widening));

%绘图
subplot(2,2,4);
plot(Et_array, k_density*N_trap_widening, 'LineWidth', 2);
hold on;
xlabel('能级 (eV)', 'FontSize', 11);
ylabel('陷阱密度 (eV^-^1 m^-^3)', 'FontSize', 11);
title('陷阱能级-陷阱密度', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([Et_min, Et_max]);