% generate sequence for each participants and balance all the conditions.
% seq= 360x9: absent 80, no spatial biased 80, spatial biased 200 with high 160 and low 40
% seq(:,1)=target reponse direction; 1 notch top; 0 notch bottom defined in
% experiment
% seq(:,2)= distractor conditions: 2 spatial biased distractor; 1 unbiased
% distractor; 0 distractor absent;
% seq(:,3)= Different (1) or same dimension(2) conditions defined later in experiment;
% seq(:,4)= distractor in the frequent region or rare region: 2 frequent
% region/1 rare region, 0 distractor absent,
% seq(:,5)= block numbers, 
% seq(:,6)=target position,
% seq(:,7)=diractor position,
% seq(:,8)=response group, % keymap=1(top=left1, down=right2); 0(top=right2, down=left1)
% seq(:,9)=target title group, % target title to right 0/ left 1


for n=1:24
    
    blk_trials = 46;
    num_block =8;
    tot_trials = blk_trials*num_block;
     p_bf = 224/368 * 23/28; % probability for spatial biased distractor in the frequent region
     p_br = 224/368 * 5/28; % probability for spatial biased distractor in the rare region
     p_nb = 80/368;  % probability for no biased distractor      
     p_nd = 64/360;     % probability for distractor-absent
    
    % define three groups
    freq_dist_reg = mod(n,2); % frequent region: 1 left (pos:8-11); 0 right (pos:6 7 12 13)
    res_group = mod(ceil(n/6),2);  % response group 1-6 (1); group 7-12 (0)...
    tar_ori = ceil(n/12)-1;  % the first 12 (0)% title right  and the last (1)% title left
    
    % set distractor propotion in each block
    for blk=1:num_block
        lseq=1:blk_trials; % 46trials/block
        lseq=lseq(randperm(length(lseq)));
        seq=zeros(blk_trials,6);
        seq(:,1)=mod(lseq,2);  % target ! or i
        seq(:,5)=mod(lseq,23); % based on proportions: 14/23 5/23 4/23 
        % spatial biased distractor 2/ no biased 1/ no distractor 0
        for t=1:blk_trials
            if seq(t,5) < 14
                seq(t,2) = 2;
            elseif seq(t,5) > 18
                seq(t,2) = 0;
            else
                seq(t,2) = 1;
            end
        end
        seq(:,3)= 0; % Different (1) or same dimension(2) conditions defined later in experiment
        
        % assign frequent or rare ratio for the spatial biased distractor:
        % 46*p_bf biased AS present in frequent region (2) /46*p_bf in rare region (1)
        as_ud = 1:blk_trials*(p_bf+p_br);
        as_ud(as_ud<=blk_trials*p_bf)=2;
        as_ud(as_ud>blk_trials*p_br)=1;
        seq(seq(:,2)==2,4)=as_ud;
        % no biased AS in frequent or rare region
        as_ud = 1:blk_trials*p_nb;
        as_ud(as_ud<=blk_trials*p_nb/2)=2;
        as_ud(as_ud>blk_trials*p_nb/2)=1;
        seq(seq(:,2)==1,4)=as_ud;
        seq(seq(:,2)==0,4)=0; % distractor-absent
        seq(:,5)=blk;
        
        blok_seq((blk-1)*blk_trials+1:blk*blk_trials,:)=seq; 
    end
    
    seq=blok_seq;
    seq = seq(randperm(size(seq,1)),:);
    
    % Add the target position. .
    pos=[6:13];
    tar_pos_eq1=repmat(pos,1,8);
    tar_pos_eq2=repmat(pos,1,5);
    tar_pos_eq3=repmat(pos,1,23);
    % distractor-absent: target positions equally appear in 6-13
    seq(seq(:,2)==0,6)=tar_pos_eq1(randperm(length(tar_pos_eq1)));
    % no spatial biased distractor
    seq(seq(:,2)==1 & seq(:,4)==1,6)=tar_pos_eq2(randperm(length(tar_pos_eq2))); %  rare
    seq(seq(:,2)==1 & seq(:,4)==2,6)=tar_pos_eq2(randperm(length(tar_pos_eq2))); %  fre
    %spatial biased distractor
    seq(seq(:,2)==2 & seq(:,4)==1,6)=tar_pos_eq2(randperm(length(tar_pos_eq2))); %when the spatial biased distractor occured in the rare. area
    seq(seq(:,2)==2 & seq(:,4)==2,6)=tar_pos_eq3(randperm(length(tar_pos_eq3))); %when the spatial biased distractor occured in the fre. area
    
    % Add the distractor position for spatial biased distractor and no
    % biased distractor
    left_pos = [8:11];
    right_pos = [12:13 6:7];
    if freq_dist_reg == 1 % biased distractor present in frequent region (1 left 0 right)
        as_ud = repmat(left_pos, 1, 46);
        nb_ud = repmat(left_pos, 1, 10);
    else
        as_ud = repmat(right_pos, 1, 46);
        nb_ud = repmat(right_pos, 1, 10);
    end
    as_ud = as_ud(randperm(length(as_ud)));
    seq(seq(:,2)==2 & seq(:,4)==2,7) = as_ud;
    seq(seq(:,2)==1 & seq(:,4)==2,7) = nb_ud(randperm(length(nb_ud)));
   % check distractor are not in the same location
   ind=[];
   ix =[];
   ind = find(seq(:,2)==2 & seq(:,4)==2);
   m=0;
   for k = 2:length(ind)
       if ind(k)-1 == ind(k-1) & seq(ind(k),7) == seq(ind(k-1),7)    % continues trials
           m=m+1;
           if m==1  % the first repeated trials
                ix = [ix  ind(k)];
                m=m+1;
           elseif ind(k) ~= ix(end)+1  % 3 or four continues trial, remove the middle one
               ix = [ix  ind(k)];
           end
       end
   end
   temp_seq= seq(ix,:); % examine same or next
    for l=1:size(temp_seq,1) 
        if temp_seq(l,7)== seq(ix(l)-1,7) %  distractor location are the same 
            j=1;
            while temp_seq(l,7) == temp_seq(j,7)    % Find a trial with a different target location
                j=j+1;
            end
            temp=temp_seq(l,7); %  
            temp_seq(l,7)=temp_seq(j,7);
            temp_seq(j,7)=temp;
        end
    end
    seq(ix,:)=temp_seq;
    
    
    if freq_dist_reg == 0   % biased distractor present in rare region (0 left 1 right)
        as_ud = repmat(left_pos, 1, 10);
        nb_ud = repmat(left_pos, 1, 10);
    else
        as_ud = repmat(right_pos, 1, 10);
        nb_ud = repmat(right_pos, 1, 10);
    end
    as_ud = as_ud(randperm(length(as_ud)));
    seq(seq(:,2)==2 & seq(:,4)==1,7) = as_ud;
    seq(seq(:,2)==1 & seq(:,4)==1,7) = nb_ud(randperm(length(nb_ud)));
    % check distractor are not in the same location
    ind=[];
    ix = [];
    ind = find(seq(:,2)==2 & seq(:,4)==1);
    m=0;
   for k = 2:length(ind)
       if ind(k)-1 == ind(k-1) & seq(ind(k),7) == seq(ind(k-1),7)    % continues trials
           m=m+1;
           if m==1  % the first repeated trials
                ix = [ix  ind(k)];
                m=m+1;
           elseif ind(k) ~= ix(end)+1  % 3 or four continues trial, remove the middle one
               ix = [ix  ind(k)];
           end
       end
   end
    temp_seq= seq(ix,:); % examine same or next
    for l=1:size(temp_seq,1) 
        if temp_seq(l,7)== seq(ix(l)-1,7) %  distractor location are the same 
            j=1;
            while temp_seq(l,7) == temp_seq(j,7) & j< size(temp_seq,1)  % Find a trial with a different target location
                j=j+1;
            end
            temp=temp_seq(l,7); %  
            temp_seq(l,7)=temp_seq(j,7);
            temp_seq(j,7)=temp;
        end
    end
    seq(ix,:)=temp_seq;
       
    % check whether target and biased distractor are in the same place or next to each other
    ind=[];
    ix = [];
    ind = find(seq(:,2)==2);
    for k = 1:length(ind)
        if isNeighbours(seq(ind(k),6),seq(ind(k),7))
            ix = [ix ind(k)];
        end
    end
    temp_seq= seq(ix,:); % examine same or next
    for i=1:size(temp_seq,1) 
        if isNeighbours(temp_seq(i,6),temp_seq(i,7)) % Target and distractor location are the same or next two each other
            j=1;
            while(isNeighbours(temp_seq(i,6),temp_seq(j,7)) || isNeighbours(temp_seq(j,6), temp_seq(i,7))) % Find a trial with a different target location
                j=j+1;
            end
            temp=temp_seq(i,6); %  
            temp_seq(i,6)=temp_seq(j,6);
            temp_seq(j,6)=temp;
        end
    end
    seq(ix,:)=temp_seq;
    

    % check whether target and no biased distractor are in the same place
    % or next to each other for no spatial biased distractor
    ind=[];
    ix = [];
    ind = find(seq(:,2)==1);
    for k = 1:length(ind)
        if isNeighbours(seq(ind(k),6),seq(ind(k), 7))
            ix = [ix ind(k)];
        end
    end
    temp_seq= seq(ix,:);
    for i=1:size(temp_seq,1) % Check the relevant sequence for cases where target and distractor location are the same or next to each other
        if isNeighbours(temp_seq(i,6),temp_seq(i,7)) % Target and distractor location are the same
            j=1;
            while(isNeighbours(temp_seq(i,6),temp_seq(j,7)) || isNeighbours(temp_seq(j,6), temp_seq(i,7))) % Find a trial with a different target location
                j=j+1;
            end
            temp=temp_seq(i,6); % swap the two target locations ?
            temp_seq(i,6)=temp_seq(j,6);
            temp_seq(j,6)=temp;
        end
    end
    seq(ix,:)=temp_seq;
    
    % assign the target oritentation (top/bottom response) equally within conditions
    % distractor-absent
    seq(seq(:,2)==0,1)=mod(randperm(64),2);
    % no spatial biased distractor
    seq(seq(:,2)==1 & seq(:,4)==1,1)=mod(randperm(40),2); 
    seq(seq(:,2)==1 & seq(:,4)==2,1)=mod(randperm(40),2);
    % spatial biased distractor
    seq(seq(:,2)==2 & seq(:,4)==1,1)=mod(randperm(40),2);
    seq(seq(:,2)==2 & seq(:,4)==2,1)=mod(randperm(184),2);
    
    % Now randomize the trial order
%     seq = seq(randperm(size(seq,1)),:);
    seq = sortrows(seq,5);
    seq(:,8)= res_group;
    seq(:,9)= tar_ori;
    save(['sequences/seq_' num2str(n) '.mat'], 'seq')
end

a= seq(seq(:,2)==0,:);
b=seq(seq(:,2)==1 & seq(:,4)==1,:);
c=seq(seq(:,2)==1 & seq(:,4)==2,:);
d=seq(seq(:,2)==2 & seq(:,4)==1,:);
e= seq(seq(:,2)==2 & seq(:,4)==2,:);
