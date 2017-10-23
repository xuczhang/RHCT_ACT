data_dir = 'D:/Dataset/RLHH/airbnb/';
file_path = strcat(data_dir, 'LA');
train_file = strcat(file_path, '_train');
test_file = strcat(file_path, '_test');

data_train = load(train_file);
Xtr_arr = data_train.Xtr_arr;
Ytr_arr = data_train.Ytr_arr;

data_test = load(test_file);
Xte_arr = data_test.Xte_arr;
Yte_arr = data_test.Yte_arr;

block_num = size(Ytr_arr, 2);
cr = 0.4;
dup_num = 10;


for idx = 1:1:dup_num

    b_finish = 0;
    while ~b_finish
        cr_v = rand(block_num, 1);
        cr_v = cr_v/sum(cr_v)*cr;

        a = cr_v(cr_v > 1/block_num);
        if size(a,1) == 0
            b_finish = 1;
        end
    end

    %disp(cr_v);
    %cr_v = [0.0625, 0, 0, 0, 0, 0, 0, 0.0625, 0.0625, 0, 0, 0, 0, 0, 0, 0.2125];
    
    Xtr = [];
    Ytr = [];
    Xte = [];
    Yte = [];
    for i=1:block_num
        
        Xtr_i = Xtr_arr{i};
        Ytr_i = Ytr_arr{i};        
        Xte_i = Xte_arr{i};
        Yte_i = Yte_arr{i};
        
        n = size(Ytr_i, 1);
        n_total = n*block_num;    
        n_o = int32(cr_v(i)*n_total); % corruption sample number(from 100 to 1200)
        n_u = n - n_o;

        %u_range = 5*norm(Ytr_i, inf);
        u_range = 100*norm(ytr_a, inf);
        u = -u_range + 2*u_range*rand(n_o,1);


        ytr_a = Ytr_i(1:n_u);
        ytr_o = Ytr_i(n_u+1:end);
        ytr_o = ytr_o + u;
        Ytr_new_i = [ytr_a; ytr_o];
        Ytr = [Ytr; Ytr_new_i];
        
        Xtr = [Xtr Xtr_i];
        
        Xte = [Xte Xte_i];
        Yte = [Yte; Yte_i];
        

    end

    output_file = strcat(file_path, '_cr', num2str(int16(cr*100)), '_', num2str(idx), '.mat'); 
    save(output_file, 'Xtr', 'Ytr', 'Xte', 'Yte');

end