    
    win = 4;
    stride = 2;
    dict_size = 5;
    
    % define folder
    folder = '../../inlabStr/subject/';
    subjectname = [subj];
    subjfolder = [folder, subjectname,'/'];
    energyfolder = [subjfolder,'feature/energy/'];
    segfolder = [subjfolder, 'segmentation/'];
%     act_rootfolder = [segfolder, 'activity/'];
    featfolder = [subjfolder, 'feature/'];
    gtHtFolder = [folder, subj,'/segmentation/rawdata_gt/'];
    
    
    
%     'Dzung': ['Spoon', 'Straw','HandFries','HandChips'],
%         'JC': ['Spoon', 'SaladFork','HandSourPatch', 'HandBread', 'HandChips'],
%         'Matt': ['HandChips', 'HandBread','HandChips', 'HandBread','Spoon', 'SaladFork', 'Cup', 'Bottle'],
%         'Jiapeng': ['HandCracker', 'Popcorn', 'HandChips','Cup', 'LicksFingers', 'Spoon','Cup','SaladFork', 'HandBread', 'SaladSpoon'],
%         'Eric': ['HandChips','Cup','Spoon'],
%         'Will':  ['popcorn', 'RedFish', 'swedishFish', 'chips', 'EatsFromBag', 'bottle', 'Bottle'], 
%         'Shibo': ['Straw', 'HandFries','HandBurger', 'Spoon'],
%         'Rawan': ['HandChips'],
%         'Cao': ['Cup'],
%     