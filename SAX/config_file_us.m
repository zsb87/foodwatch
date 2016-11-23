    
    win = 4;
    stride = 2;
    dict_size = 10;

    % define folder
    folder = '../../inlabUnstr/subject/';
    subjectname = [subj,'(8Hz)'];
    subjfolder = [folder, subjectname,'/'];
    energyfolder = [subjfolder,'feature/energy/'];
    segfolder = [subjfolder, 'segmentation/'];
    act_rootfolder = [segfolder, 'activity/'];
    featfolder = [subjfolder, 'feature/'];
    
    activities = { 
        'answerPhone';           
        'bottle';     
        'brushTeeth';
        'cheek';        
        'chips';        
        'chopsticks';     
        'comb';            
        'cup';
        'drinkStraw';        
        'forehead';
        'fork';
        'knifeFork';
        'nose'; 
        'pizza';
        'restNearMouth';
        'Phone';    
        'smoke_im'; 
        'smoke_ti';
        'soupSpoon';
    };

    