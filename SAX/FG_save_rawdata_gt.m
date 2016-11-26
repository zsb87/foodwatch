
function FG_save_rawdata_gt(subj, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file!')
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % save all activities' ground truth in 'gt_feeding_headtail.csv'
    %   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    testdata_labeled = [folder, subj,'/testdata_labeled.csv'];
    data = csvread(testdata_labeled,1,1);
    
    fClass = data(:,raw_fCol);
    nfClass = data(:,raw_nfCol);
    if ~exist(gtHtFolder, 'dir')    mkdir(gtHtFolder),   end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save all activities' ground truth in one file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    f_gt_headtail = pointwise2headtail(fClass);
    save_headtail(f_gt_headtail, strcat(gtHtFolder,'gt_feeding_headtail.csv'));
    
    nf_gt_headtail = pointwise2headtail(nfClass);
    save_headtail(nf_gt_headtail, strcat(gtHtFolder,'gt_nonfeeding_headtail.csv'));
end


