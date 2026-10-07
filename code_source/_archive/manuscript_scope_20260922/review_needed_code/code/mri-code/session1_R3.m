function as_session1

blk_trials = 23;
num_block = 8;
tot_trials = blk_trials*num_block;

exp = CExp(num_block, blk_trials,'blockFactors',1,'blockRepetition',1); %list of trials
exp.subInfo('group','1');
exp.sName = [exp.sName '_s13'];
exp.seq=zeros(tot_trials,9);
load(['sequences/seq_' num2str(exp.sPara) '.mat']);
exp.seq=seq(1:184,:); %load three blocks
exp.maxTrls = size(exp.seq,1);
exp.seq(:,3)=1; % Different dimension conditionss
popSearch(exp);
ShowCursor();
end
