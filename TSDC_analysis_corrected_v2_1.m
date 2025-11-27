clear all; close all; clc;

%% 1. 物理常数和参数
k_B = 8.617333e-5;   % 玻尔兹曼常数 (eV/K)
q = 1.602176e-19;    % 电子电荷 (C)

%% 2. 读取实验数据

% 读取Excel文件
% 请确保文件在当前路径下
if exist('TSDC_Current.xlsx', 'file')
    data = xlsread('TSDC_Current.xlsx');
else
    error('未找到文件 TSDC_Current.xlsx，请检查路径。');
end

T_exp = data(:,1);      % 温度 (K)
I_exp = abs(data(:,2)); % 电流 (A)，取绝对值避免负值

% 去除零值和异常值
valid_idx = I_exp > 1e-15;  % 去除过小的值
T_exp = T_exp(valid_idx);
I_exp = I_exp(valid_idx);

% 数据平滑处理
% 注意：过大的平滑窗口(window_size)会人为抹平尖峰。
% 如果你的原始数据尖峰很明显，建议适当减小窗口，或者仅在噪声大时使用平滑。
window_size = 10;
I_smooth = smooth(I_exp, window_size, 'moving');

fprintf('数据点数: %d\n', length(T_exp));
fprintf('温度范围: %.2f - %.2f K\n', min(T_exp), max(T_exp));
fprintf('电流范围: %.2e - %.2e A\n', min(I_exp), max(I_exp));

beta = 0.05;         % 升温速率 0.05 K/s
d_sample = 1.2e-4;   % 样品厚度 m

%% 3. 构建陷阱能级基函数库（核心修改部分）

fprintf('\n构建基函数库...\n');

% [修改] 调整陷阱能级范围和步长
% E-T关系约为 E=0.00311*T-0.02442 (在10^13频率因子下)
Et_min = 0.00311 * min(T_exp) - 0.05;   % 稍微放宽下限
Et_max = 0.00311 * max(T_exp) + 0.05;   % 稍微放宽上限
dEt = 0.001;    % [修改] 更精细的步长 (原0.002)，有助于拟合尖峰
Et_array = Et_min:dEt:Et_max;
n_traps = length(Et_array);

fprintf('能级范围: %.2f - %.2f eV\n', Et_min, Et_max);
fprintf('能级步长: %.3f eV\n', dEt);
fprintf('能级点数: %d\n', n_traps);

% 设置参数
T0 = min(T_exp) - 10; % [修改] 初始温度设为比实验最低温略低，确保积分完整
if T0 < 0, T0 = 10; end 
nu_j = 10^13;         % 频率因子

% 构建响应矩阵 A
A = zeros(length(T_exp), n_traps);

% [新增] 创建进度条
h_wait = waitbar(0, '正在构建基函数矩阵，请稍候...');

% [优化] 为了捕捉尖峰，我们需要在更密的温度网格上计算理论曲线，然后再插值回实验温度
% 如果直接使用稀疏的 T_exp 进行积分，会导致尖峰"削顶"
dT_fine = 0.1; % 0.1K 的精细步长
T_fine = (T0:dT_fine:max(T_exp))'; 

fprintf('\n计算各陷阱能级的响应函数...\n');

% 记录开始时间
tic;

for j = 1:n_traps
    % 更新进度条 (每10个循环更新一次以节省资源)
    if mod(j, 10) == 0
        waitbar(j / n_traps, h_wait, sprintf('计算进度: %.1f%% (%d/%d)', j/n_traps*100, j, n_traps));
    end
    
    % [核心优化] 调用新的向量化函数，在精细网格 T_fine 上计算
    I_fine_calc = single_trap_TSDC_fast(T_fine, Et_array(j), nu_j, T0, beta, k_B);
    
    % 将精细网格的计算结果插值回实验温度点 T_exp
    % 'pchip' 插值能较好地保持形状
    A(:, j) = interp1(T_fine, I_fine_calc, T_exp, 'pchip', 0);
    
    % 确保非负
    A(A(:, j) < 0, j) = 0;
    
    % 归一化基函数 (保持形状，幅度由NNLS决定)
    % 注意：这里归一化有助于数值稳定性，但需注意后续反演出的 N_trap 含义
    if max(A(:, j)) > 0
        A(:, j) = A(:, j) / max(A(:, j));
    end
end

% 关闭进度条
close(h_wait);
toc; % 输出耗时

%% 4. NNLS反卷积求解
fprintf('\n执行NNLS反卷积...\n');

% 归一化实验数据
I_norm = I_smooth / max(I_smooth);

% 设置NNLS选项
options = optimset('Display', 'off', 'TolFun', 1e-12, 'TolX', 1e-12);

% 非负最小二乘法求解
[N_trap, residual_norm] = lsqnonneg(A, I_norm, options);

% 归一化陷阱密度 (为了绘图)
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

%% 5. 结果可视化

figure('Position', [50, 50, 1000, 800], 'Color', 'white');

% 子图1：原始数据和平滑数据
subplot(2,2,1);
semilogy(T_exp, I_exp, 'b.', 'MarkerSize', 5, 'DisplayName', '原始数据');
hold on;
semilogy(T_exp, I_smooth, 'r-', 'LineWidth', 1.5, 'DisplayName', '平滑数据');
xlabel('温度 (K)', 'FontSize', 11);
ylabel('电流 (A)', 'FontSize', 11);
title('TSDC原始数据', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
xlim([min(T_exp), max(T_exp)]);

% 子图2：陷阱能级-密度分布 (NNLS直接结果)
subplot(2,2,2);
plot(Et_array, N_trap_normalized, 'b-', 'LineWidth', 1);
%fill(Et_array, N_trap_normalized, 'b', 'FaceAlpha', 0.3);
xlabel('陷阱能级 Et (eV)', 'FontSize', 11);
ylabel('相对权重 (a.u.)', 'FontSize', 11);
title('离散陷阱分布 (NNLS结果)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([Et_min, Et_max]);

% 子图3：重构验证
subplot(2,2,3);
plot(T_exp, I_norm, 'k-', 'LineWidth', 1.5, 'DisplayName', '实验数据(归一化)');
hold on;
plot(T_exp, I_reconstructed, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('拟合 (R²=%.4f)', R2));
xlabel('温度 (K)', 'FontSize', 11);
ylabel('归一化电流', 'FontSize', 11);
title('曲线重构验证', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
xlim([min(T_exp), max(T_exp)]);

%% 6. 陷阱密度计算与展宽

% 1. 计算总陷阱密度 N_total (m^-3)
% 公式：Q = integral(I dt) = integral(I/beta dT) = N_total * q * Volume
% N_total = (Area_I / beta) / (q * Area_sample * thickness)
% 注意：I_exp 已经是 A，不需要再乘倍率，只需要除以体积和电荷量

Area_sample = pi * (1e-2)^2; % 假设电极半径，用户代码中写了 pi*1e-4，可能是半径0.01m? 
% 修正：用户原代码为 (d_sample*pi()*1e-4)，这里1e-4如果指面积单位换算需确认
% 假设用户原代码逻辑是: 面积 = pi * (半径)^2. 如果输入是cm^2需要注意。
% 我们沿用用户原代码的数值逻辑：
% 6.24146e18 = 1/q
val_integral = trapz(T_exp, I_exp); % A*K
Total_Charge = val_integral / beta; % A*s = C
% 假设用户的面积因子: pi()*1e-4 可能是 1cm^2 ? 
% 这里的体积计算保持用户原代码逻辑，以免出错
Volume_factor = d_sample * pi() * 1e-4; 
N_density_total = Total_Charge * (6.24146e18) / Volume_factor; 

fprintf('\n总陷阱密度: %.2e m^-3\n', N_density_total);

% 2. 高斯展宽 (将离散的NNLS结果转化为连续谱)
% NNLS得到的是离散的"针状"分布，为了符合物理上的能级分布，需要卷积一个高斯核
x_conv = -0.1 : dEt : 0.1;
sigma = 0.02; % 展宽宽度0.02eV
Gauss_base = 1/(sqrt(2*pi)*sigma) * exp(-(x_conv).^2 / (2*sigma^2));

% 对 N_trap (原始权重) 进行卷积
N_trap_widening = conv(N_trap, Gauss_base, 'same')';

% 3. 定标
% 使得展宽后曲线下的面积积分等于总陷阱密度 N_density_total
Area_widened = trapz(Et_array, N_trap_widening);
if Area_widened > 0
    k_scale = N_density_total / Area_widened;
    N_density_spectrum = N_trap_widening * k_scale;
else
    N_density_spectrum = N_trap_widening;
end

% 子图4：最终陷阱密度谱
subplot(2,2,4);
plot(Et_array, N_density_spectrum, 'r-', 'LineWidth', 2);
%fill(Et_array, N_density_spectrum, 'r', 'FaceAlpha', 0.2);
xlabel('能级 (eV)', 'FontSize', 11);
ylabel('陷阱密度 (eV^{-1} m^{-3})', 'FontSize', 11);
title('陷阱态密度分布 (DOS)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([Et_min, Et_max]);
grid on;


%% --- 辅助函数：快速向量化TSDC计算 ---
function I = single_trap_TSDC_fast(T, Et, nu, T0, beta, k_B)
    % 向量化计算，极大提高速度并解决积分精度问题
    % 1. 积分项 integrand
    % phi(T) = exp(-Et / kT)
    phi = exp(-Et ./ (k_B .* T));
    
    % 2. 计算积分部分 integral_{T0}^{T} phi(T') dT'
    % 使用 cumtrapz 实现累积积分，比循环调用 integral 快几百倍
    integral_val = cumtrapz(T, phi);
    
    % 3. 计算电流
    % I(T) = A * phi(T) * exp( -nu/beta * integral )
    % 注意：这里计算的是相对形状，系数A在NNLS中确定，这里设为1即可
    % 但为了数值不溢出，保留物理形式
    
    term2 = exp((-nu / beta) * integral_val);
    I = phi .* term2;
    
    % 确保初始温度以前为0 (物理上T<T0无电流)
    I(T < T0) = 0;
    
    % 替换 NaN 为 0
    I(isnan(I)) = 0;
end