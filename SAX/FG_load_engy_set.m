function [test_sig_cell, test_gt_global_htcell, test_gt_local_htcell, train_sig_cell, train_gt_htcell] = FG_load_engy_set(test_subj, train_subj, config_file)
            %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!_seg_detect')
    end
    
    win = 4;
    stride = 2;
    dict_size = 5;
    
    
    sigfolder = [folder, train_subj,'/feature/energy/'];
    sigfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(strcat(sigfolder,sigfile),1);
    
    train_sig_cell{1} = data(:,2);
    train_gt_htcell{1} = csvread([folder, train_subj,'/segmentation/engy_gt/gt_feeding_headtail.csv']);% it has (n,3) shape now
    
    
    
    folder = '../../inlabStr/subject/';
    
    sigfolder = [folder, test_subj,'/feature/energy/'];
    sigfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(strcat(sigfolder,sigfile),1);
    
    test_sig_cell{1} = data(:,2);
    test_gt_global_htcell{1} = csvread([folder, test_subj,'/segmentation/engy_gt/gt_feeding_headtail.csv']);% it has (n,3) shape now
    test_gt_local_htcell =  test_gt_global_htcell;
   
end