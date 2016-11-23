function [motif_SAX_cell] = FG_KcentriodSC_27(train_sig_cell, train_gt_htcell, nn, dict_size)
        
    motif_SAX_cell = [];
    
    train_HT = [];    
    for i = 1:size(train_gt_htcell,2)
        train_HT = [train_HT; train_gt_htcell{i}];
    end
    
    
    X = [];    
   
%     for i = 1:size(train_HT,1)
%         maxlen = max(train_HT(:,3));
%         x =  train_sig_cell{i}(train_HT(i,1):train_HT(i,2));
%         X = [X;zeros(1,maxlen - train_HT(i,3)),x'];
%     end

    for i = 1:size(train_HT,1)
        maxlen = max(train_HT(:,3));
%         x =  train_sig_cell{1}(train_HT(i,1):train_HT(i,2));
        x{i} =  train_sig_cell{1}(train_HT(i,1):train_HT(i,2));
        X = [X;zeros(1,maxlen - train_HT(i,3)),x{i}'];
    end
        
    [cluster_num, cent] = ksc_toy(X, nn);
    cent(all(cent==0,2),:)=[];
        
        
    mcell_ind = 1;
    
    for i = 1:size(cent,1)
            
        c_i = cent(i,:); % c_i means center_i
        % remove NaN, otherwise error msg when dealing with Shibo
        ind = find(isnan(c_i));
        c_i(ind)=0;    
        c_i_wo0 = c_i(find(any(c_i,1),1,'first'):end);
        
        motif_SAX_cell{mcell_ind} = timeseries2symbol(c_i_wo0, length(c_i_wo0), floor(length(c_i_wo0)/2), dict_size,1);
            
        mcell_ind = mcell_ind + 1;
        
    end
    
    mcell_ind = mcell_ind - 1;
        
    ind_tmp = 1;
        
    for i = 1:mcell_ind
        if all(motif_SAX_cell{i}==0,2)==0            
            ptns_cell_tmp{ind_tmp} = motif_SAX_cell{i};
            ind_tmp = ind_tmp + 1;
        end
    end
    
    motif_SAX_cell = ptns_cell_tmp;
    
end
