function as_session2

blk_trials = 46;
num_block = 4;
tot_trials = blk_trials*num_block;

exp = CExp(num_block, blk_trials,'blockFactors',1,'blockRepetition',1); %list of trials
exp.subInfo('group','1');  
exp.sName = [exp.sName '_s24'];
exp.seq=zeros(tot_trials,9);
load(['sequences/seq_' num2str(exp.sPara) '.mat']);
exp.seq=seq(185:end,:); %another three blocks
exp.maxTrls = size(exp.seq,1);
exp.seq(:,3)=2; % same dimension conditionss
popSearch(exp);
ShowCursor();
end
