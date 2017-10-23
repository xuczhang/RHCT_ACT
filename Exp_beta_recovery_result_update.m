p = 400;
k = 4;
bNoise = 0;
%obj_num = 5;
noise_str = '';
if ~bNoise
   noise_str = 'nn_'; 
end

%result_path = 'D:/Dropbox/PHD/publications/IJCAI2017_RLHH/experiment/';
result_path = FindResultPath();
data_file = strcat(result_path, 'beta_', num2str(k), 'K_', 'p', num2str(p), '_', noise_str);
data_file = data_file(1:end-1);
data_file = strcat(data_file, '.mat');
data = load(data_file);

OLS_result = data.OLS_result;
DALM_result = data.DALM_result;
HOMO_result = data.HOMO_result;
TORRENT0_result = data.TORRENT0_result;
TORRENT25_result = data.TORRENT25_result;
TORRENT50_result = data.TORRENT50_result;
RLHH_result = data.RLHH_result;


%DALM_result=[1.847957432507405E-5	5.8519841721511684E-5	7.837887543346209E-5	1.0608551271958273E-4	6.645094278305013E-5	2.761805567084911E-5	3.4133405430352105E-5	2.46586778082317E-5];
RACT_result=[3.000304207812699E-4	1.813968457779318E-15	7.436104950271494E-5	1.6191254717135023E-4	9.932867672923363E-4	3.268875304219479E-4	3.381892141718182E-4	0.0012359207597530887];

save(data_file, 'OLS_result', 'DALM_result', 'HOMO_result', 'TORRENT0_result', 'TORRENT25_result', 'TORRENT50_result', 'RLHH_result', 'RACT_result');
