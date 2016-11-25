
    win = 4;
    stride = 2;
    dict_size = 5;

    % define folder
    folder = '../../inlabUnstr/subject/';
    subjectname = [subj];
    subjfolder = [folder, subjectname,'/'];
    energyfolder = [subjfolder,'feature/energy/'];
    segfolder = [subjfolder, 'segmentation/'];
    featfolder = [subjfolder, 'feature/'];
    gtHtFolder = [folder, subj,'/segmentation/rawdata_gt/'];
%     
%     activities = { 
%         'answerPhone';           
%         'bottle';     
%         'brushTeeth';
%         'cheek';        
%         'chips';        
%         'chopsticks';     
%         'comb';            
%         'cup';
%         'drinkStraw';        
%         'forehead';
%         'fork';
%         'knifeFork';
%         'nose'; 
%         'pizza';
%         'restNearMouth';
%         'Phone';    
%         'smoke_im'; 
%         'smoke_ti';
%         'soupSpoon';
%     };

    