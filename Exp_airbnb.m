data_dir = 'D:/Dataset/RLHH/airbnb/';
data_type = 'NY';
cr = 0.4;
%train_file = strcat(data_dir, 'NY_train_cr40');
dup_num = 5;

OLS_err = [];
DALM_err = [];
HOMO_err = [];
TORRENT0_err = [];
TORRENT25_err = [];
TORRENT50_err = [];
RLHH_err = [];
RACT_err = [];

test_file = strcat(data_dir, data_type, '_test');
data = load(test_file);
Xte_arr = data.Xte_arr;
Yte_arr = data.Yte_arr;

fprintf('=== cr:[%g] ===\n', cr);
for idx = 1:1:dup_num

    fprintf('=== [%g] ===\n', idx);
    
    file = strcat(data_dir, data_type, '_cr', num2str(int32(cr*100)), '_', num2str(idx), '.mat');
    data = load(file);
    Xtr = data.Xtr;
    Ytr = data.Ytr;
    Xte = data.Xte;
    Yte = data.Yte;
    

    % OLS: Ordinary Least Square 
%     OLS_beta = regress(Ytr, Xtr');
%     OLS_err = [OLS_err Metrics_MSE(Xte, Yte, OLS_beta)];
    
    % DALM Method
    start = 10001;
    ds = 10000;
    STOPPING_TIME = -2;
    beta_truth = OLS_beta;
    maxTime = 8;
    DALM_beta = Baseline_DALM_CBM(Xtr(:,start:ds+start-1)', Ytr(start:ds+start-1), 'stoppingCriterion', STOPPING_TIME, 'groundtruth', beta_truth, 'maxtime', maxTime, 'maxiteration', 1e6);
    DALM_err = [DALM_err Metrics_MSE(Xte, Yte, DALM_beta)];

    % Homotopy Method
    STOPPING_TIME = -2;
    beta_truth = OLS_beta;
    HOMO_beta = Baseline_Homotopy_CBM(Xtr(:,start:ds+start-1)', Ytr(start:ds+start-1), 'stoppingCriterion', STOPPING_TIME, 'groundtruth', beta_truth, 'maxtime', maxTime, 'maxiteration', 1e6);
    HOMO_err = [HOMO_err Metrics_MSE(Xte, Yte, HOMO_beta)];


    % TORRENT0
%     TORRENT0_beta = Baseline_TORRENT( Xtr, Ytr, cr);
%     TORRENT0_err = [TORRENT0_err Metrics_MSE(Xte, Yte, TORRENT0_beta)]; 
% 
%   
%     % TORRENT25
%     TORRENT25_beta = Baseline_TORRENT( Xtr, Ytr, cr/1.25);
%     TORRENT25_err = [TORRENT25_err Metrics_MSE(Xte, Yte, TORRENT25_beta)]; 
% 
%     % TORRENT50
%     TORRENT50_beta = Baseline_TORRENT( Xtr, Ytr, cr/1.5);
%     TORRENT50_err = [TORRENT50_err Metrics_MSE(Xte, Yte, TORRENT50_beta)]; 
%     
%     % RLHH
%     [RLHH_beta, S] = RLHH(Xtr, Ytr);
%     RLHH_err = [RLHH_err Metrics_MSE(Xte, Yte, RLHH_beta)];     
% 
%     % RACT
%     [RACT_beta, S] = RACT(Xtr, Ytr);
%     RACT_err = [RACT_err Metrics_MSE(Xte, Yte, RACT_beta)];     

end

%fprintf('OLS_mse:[%f] RLHH_mse:[%f] OPAA_mse:[%f] ORL_mse:[%f] ORL0_mse:[%f] BatchRC_mse:[%f] OnlineRC_pc:[%f]\n', OLS_err/dup_num, RLHH_err/dup_num, OPAA_err/dup_num, ORL_err/dup_num, ORL0_err/dup_num, BatchRC_err/dup_num, OnlineRC_err/dup_num);
fprintf('OLS mean %.3f\t%.3f\n', mean(OLS_err), std(OLS_err));
fprintf('DALM mean %.3f\t%.3f\n', mean(DALM_err), std(DALM_err));
fprintf('HOMO mean %.3f\t%.3f\n', mean(HOMO_err), std(HOMO_err));
fprintf('TORRENT0 mean %.3f\t%.3f\n', mean(TORRENT0_err), std(TORRENT0_err));
fprintf('TORRENT25 mean %.3f\t%.3f\n', mean(TORRENT25_err), std(TORRENT25_err));
fprintf('TORRENT50 mean %.3f\t%.3f\n', mean(TORRENT50_err), std(TORRENT50_err));
fprintf('RLHH mean %.3f\t%.3f\n', mean(RLHH_err), std(RLHH_err));
fprintf('RACT mean %.3f\t%.3f\n', mean(RACT_err), std(RACT_err));

