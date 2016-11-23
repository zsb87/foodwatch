subj = 'Dzung';    
win = 4;
    stride = 2;
    dict_size = 10;
    
    if strcmp(subj, 'Rawan')
        activities = { 
        'Spoon';            %1
        'HandBread';        %2
        'Chopstick';        %3
        'KnifeFork';        %4
        'SaladFork';        %5
        'HandChips';        %6
        'Cup';              %7
        'Straw';            %8
        'Phone';            %9
        'SmokeMiddle';      %10
        'SmokeThumb';       %11
        'Bottle';           %12
        'Nose';             %13
        'ChinRest';         %14
        'Scratches';        %15
        'Mirror';           %16
        'Teeth';            %17
        };
    else
        activities = { 
        'Spoon';            %1
        'HandBread';        %2
        'Cup';              %3
        'Chopstick';        %4
        'KnifeFork';        %5
        'Bottle';           %6
        'SaladFork';        %7
        'HandChips';        %8
        'Straw';            %9
        'SmokeMiddle';      %10
        'SmokeThumb';       %11
        'ChinRest';         %12
        'Phone';            %13
        'Mirror';           %14
        'Scratches';        %15
        'Nose';             %16
        'Teeth';            %17
        };
    end
    

    % define folder
    folder = '../../inlabStr/subject/';
    subjectname = [subj,'(8Hz)'];
    subjfolder = [folder, subjectname,'/'];
    featfolder = [subjfolder,'feature/'];
    actfolder  = [subjfolder, 'segmentation/activity/'];
    %% read energy file
    %% check the column number
    energyfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_feedinglike_labeled.csv'];
    data = csvread(strcat(featfolder,energyfile),1,1);
    Class = data(:,9);  
    Class = Class(1:2:end);  
%     activity = data(:,9);

    % save ground truth for all activities in one file
    gt_headtail = pointwise2headtail(Class);
    gtfile_all = 'gt_headtail_feedinglike_activities.csv';
    save_headtail(gt_headtail, strcat(actfolder,gtfile_all));

