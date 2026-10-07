function popSearchBe(exp)
% Additional singleton task.
% Replication of Rene 2020 + fMRI study
% If withMRI, 4 criutial codes should be changed because of the mirror, including:
% kbinput/ target title orientation/ x-axis coordinates /display text/
% Bei 04/2021

try
    withMRI = 1; % please change it to 1 with MRI machine
    withEyeLink = 1; % please change to 1 with eyeLink device
    blk_trials = 23;
    num_block = 8;
    
    HideCursor();
    %%%%%%%%%%%%%%%%%% Parallel port key%%%%%%%%%%%%%%%%%%%%
    % define input device
    if withMRI  % response input
        kb = CInput('v');
    else
        kb = CInput('k',[1,2],{'f', 'j'}); % left 1 right 2
    end
    
    kb2 = CInput('k',[1,2],{'c', 's'}); % trigger input
    
    keymap= exp.seq(1, 8); % response key mapping
    dim_cond = exp.seq(1, 3); % 1 color biased dimension  2 orientation biased dimension
    tar_ori =  exp.seq(1, 9); % target title to right 0/ left 1
    switch tar_ori
        case 0 % title right
            infoText = init_text1(keymap); %instruction as follows: keymap=1(top=left1, down=right2); 0(top=right2, down=left1)
        case 1 %  title left
            infoText = init_text2(keymap); %instruction as follows
    end
    
    if withMRI
        tar_ori =  abs(tar_ori-1);  % swap the target title orientation because of the MRI mirror
    end
    
    %%%%%%%%%%%%% %visual Display%%%%%%%%%%%%%%%%%%%vs
    % creat visual display (monitor properties)
    v = DDisplay('lineSpace',2,'skipSync',0, 'fontSize', 20, ...
        'lineWidth',60,'viewDistance',140,'monitorSize', 25.3, 'bgColor',[0 0 0],'fullWindow',1);
    s
    % scanner distance: 89scanner+20scanner-board+15mirror-head+16mirror-eye=140
    % monitor size 64.3cm =25.3 inches
    
    %create vertical line (here tips: 0.81 in degree is length; 0.18 is width, that is x/y swap somehome)
    red = [255 33 51];
    turquoise = [0, 140, 209];
    grey = [128, 128, 128];
    tex_disp = v.createShape('i_shape',0.18, 0.81,'color',turquoise, 'break', [0.25 0.45]); % tips:Mac0.45,win0.35
    tex_dist = v.createShape('i_shape',0.18, 0.81,'color',red, 'break', [0.25 0.45]); % tips:Mac0.45,win0.35
    boxsize= 0.81 * sqrt(pi/4);
    tex_sq = v.createShape('rectangle', boxsize, boxsize, 'fill',0, 'color', grey);
    %     circle_diam = 0.81;
    %     tex_sq = v.createShape('circle', circle_diam, circle_diam, 'fill',0,'color', grey);
    itemSizes = [0.18 0.81]; % reverse the xy
    
    % generate xy positions (circular)
    xy(1,:) = [0 0]; %center
    num_items = [4 8 14];
    circle_rot =  [0 pi/8 0];
    icount = 1;
    for i=1:length(num_items)
        ecc = i*1.25;  % degree of visual angel for the circle
        for k = 1:num_items(i)
            icount = icount + 1;
            theta = 2*pi/num_items(i)*(k-1) + circle_rot(i);
            xy(icount, :) = ecc*[cos(theta), sin(theta)];
        end
    end
    
    % mirror the coordinates
    if withMRI
        xy1(:,1)=xy(:,1)*(-1);
        xy1(:,2)=xy(:,2);
        xy=xy1;
    end
    
    %%%%%%%%%%%%% % ITI seq and fixation seq%%%%%%%%%%%%%%%%%%
    iti=1:exp.maxTrls;
    iti=iti(randperm(length(iti)));
    itiseq=mod(iti,2)*1+3; % 3/4
    fixd=1:exp.maxTrls;
    fixseq=fixd(randperm(length(fixd)));
    for m=1:exp.maxTrls
        if  fixseq(m) <=exp.maxTrls/2  % 50% with short presentation
            fixseq(m)=0.5;
        else
            fixseq(m)=5.5;
        end
    end
    
    %%%%%%%%%%%%%% EyeLink trigger%%%%%%%%%%%%%%%%%%%%
    if withEyeLink
        % Initialize eyelink
        if ~EyelinkInit([], 1)
            Eyelink('Initialize');
        end
        
        Eyelink('OpenFile',[exp.sName '.edf']);
        el = EyelinkInitDefaults(v.wnd);
        el.backgroundcolour = 0;
        el.calibrationtargetcolour = 255;
        el.msgfontcolour=255;
        EyelinkUpdateDefaults(el)
        if ~EyelinkInit([], 1)
            fprintf('Eyelink Init aborted.\n');
            return;
        end
    end
    
    HideCursor();
    
    if withEyeLink
        % Eyelink calibration
        EyelinkDoTrackerSetup(el);
    end
    
    %%%%%%%%%%%%%% MRI trigger%%%%%%%%%%%%%%%%%%%%
    deviceIndex = [];
    keysOfInterest = zeros(1, 256);
    keysOfInterest(KbName('s')) = 1;
    KbQueueCreate(deviceIndex, keysOfInterest);
    KbQueueStart(deviceIndex);
    KbEventFlush(deviceIndex);
    % wait for a keypress
    count=1;
    trigger=[];
    
    %%%%%%%%%%%%%%formal experiment%%%%%%%%%%%%%%%%%%%%%%%%
    % display instruction
    if withMRI
        v.dispMirrorText(infoText.instruction);
    else
        v.dispText(infoText.instruction);
    end
    
    %This is where we have trigger identification code
    %     k = kb2.response(1);
    %     if (k==2)
    while KbEventAvail(deviceIndex)==0
        WaitSecs('Yieldsecs', 0.001);  % wait for 1 msec to prevent system overload
    end
    % receive the first trigger
    while KbEventAvail(deviceIndex)
        key_ev = KbEventGet(deviceIndex);
        if (~isempty(key_ev) && key_ev.Pressed)
            trigger(count)=key_ev.Time;
            count=count+1;
            key_ev=[];
        end
        WaitSecs('Yieldsecs', 0.001);  % wait for 1 msec to prevent system overload
    end
    
    % now start the main experiment with fixtion
    triggertime=GetSecs;
    while GetSecs - triggertime < 6
        v.dispFixation(20,2);
    end
    v.flip(1);
    
    %%%%%%%%%%%%%%prepare stimuli at given condition(s)%%%%%%%%%%%%%%
    % get condition for each trial and start a trial
    xy(1,:)=[]; % remove the item in the center relace with cross
    % first make fixation display
    boxSizes = [boxsize boxsize];
    box_fix = ones(1,length(xy)) *tex_sq;
    %  boxSizes =[0.18 0.81];
    for iTrl=1:exp.maxTrls
        
        % check trigger
        while KbEventAvail(deviceIndex)
            key_ev = KbEventGet(deviceIndex);
            if (~isempty(key_ev) && key_ev.Pressed)
                trigger(count)=key_ev.Time;
                count=count+1;
                key_ev=[];
            end
            WaitSecs('Yieldsecs', 0.001);  % wait for 1 msec to prevent system overload
        end
        
        fixdur= fixseq(iTrl);
        itidur= itiseq(iTrl);
        cond = exp.getCondition; %get condition array
        c_updown = cond(1); % target ! or i
        c_AS = cond(2); % additional singleton condition 2/1/0
        
        % Randomize rotations of all items (0/180) in the search display: ! or i
        items = ones(1,length(xy))*tex_disp;
        rotations = zeros(1,length(xy));
        rotations(1:floor(length(xy)/2))=180;
        rotations=rotations(randperm(length(rotations)));
        
        % set target position and it's rotations
        c_target_pos = cond(6)-1; % pos-1 for remove the center item
        if(tar_ori==0)   % title right (cond(1)==1 notch top; 0 notch bottom)
            rotations(c_target_pos) = 180*abs(c_updown-1)+12;
        else %  title left  (cond(1)==1 notch top; 0 notch bottom)
            rotations(c_target_pos) = 180*(c_updown+1)-12;
        end
        
        % set distractor position and it's rotations
        c_AS_pos=cond(7)-1; % pos-1 for remove the center item
        c_orient = ceil(rand*2)-1; % 0 or 1
        if c_AS > 0
            switch dim_cond
                case 1 % color dimension biased
                    if (c_AS==2)  % color distractor
                        items(c_AS_pos) = tex_dist;
                        rotations(c_AS_pos)=180*c_orient;
                    else  % orientation distractor
                        if (tar_ori==0)  % target title to RIGHT, dist for LEFT
                            rotations(c_AS_pos) = 180*(c_orient+1)-45;
                        else
                            rotations(c_AS_pos) = 180*c_orient+45; % target title to LEFT, dist for RIGHT
                        end
                    end
                case 2 % orientation dimension biased
                    if (c_AS ==2)  % ori distractor
                        if (tar_ori==0)
                            rotations(c_AS_pos) = 180*(c_orient+1)-45; % target title to right, dist for left
                        else
                            rotations(c_AS_pos) = 180*c_orient+45; % dist for right
                        end
                    else    % color distractor
                        items(c_AS_pos) = tex_dist;
                        rotations(c_AS_pos)=180*c_orient;
                    end
            end
        end
        
        %%%%%%%%%%%%%%start stimuli at given condition(s)%%%%%%%%%%%%%%
        
        %-------1.fixation------- %
        % make fixation array
        v.dispFixation(20,1,0);
        v.dispItems(xy, box_fix, boxSizes, 0, 0);
        if withEyeLink
            Eyelink('StartRecording'); % eyelink recording
            Eyelink('Message', 'Start of trial fixation'); % Catch target message
        end
        v.flip;
        t0 = GetSecs;
        WaitSecs(0.5);
        %  fixation cross
        while GetSecs - t0 < fixdur+0.5
            v.dispFixation(20,1);
        end
        
        %-------2. main search display stimuli------- %
        v.dispFixation(20, 1,0); % draw the fixation 0 means not flip
        v.dispItems(xy, items, itemSizes, rotations, 0);  % draw search display
        v.flip;  % show the search display
        disTime = GetSecs;
        if withEyeLink
            Eyelink('Message', 'Search display onset'); % Catch target message
        end
        
        [key, rTime] = kb.response(0);   %acquire response
        
        %------ -3. acquire response------- %
        while GetSecs - disTime <  1              % stimuli presentation time
            if key < 0 % no response yet
                [key, rTime] = kb.response(0);          %acquire response
            end
        end
        v.flip(1); % blank the screen
        
        if withEyeLink
            Eyelink('Message', 'End of trial')
            Eyelink('Stoprecording'); % eyelink stop recording
        end
        
        while GetSecs - disTime < 1.9
            if key < 0 % no response yet
                [key, rTime] = kb.response(0);          %acquire response
            else
                break;
            end
        end
        
        
        %-------4. feedback------- %
        if key==-1  % no response/too slow
            correct=0;
            if withMRI
                v.dispMirrorText('Error!');
            else
                v.dispText('Error!');
            end
            WaitSecs(0.1);
        elseif (keymap==0 && ~((c_updown+1) == key))
            correct=0;
            if withMRI
                v.dispMirrorText('Error!');
            else
                v.dispText('Error!');
            end
            WaitSecs(2 - (GetSecs-disTime));
        elseif (keymap==1 && ~(abs(c_updown-2) == key))
            correct=0;
            if withMRI
                v.dispMirrorText('Error!');
            else
                v.dispText('Error!');
            end
            WaitSecs(2 - (GetSecs-disTime));
        else
            correct = 1;  % response correct
            WaitSecs(2 - (GetSecs-disTime));
        end
        v.flip(1);
        tendTime=GetSecs;
        v.color=[255 255 255];
        
        % add disTime and valid of fixation at end of the response
        % onset time is useful for aligning the time with fMRI data
        exp.setResp([c_orient, key, correct, t0, disTime,rTime,rTime-disTime, tendTime,triggertime]); %store response: target position, key, RT
        
        %-------5. inter-trial interval------- %
        while GetSecs - tendTime < itidur
            v.flip(1);
        end
        
        
        %-------6. block break or sometime feedback------- %
        if mod(iTrl,blk_trials) == 0 && iTrl < exp.maxTrls
            blkCorrect=mean(exp.resp((iTrl-blk_trials+1):iTrl,3))*100;
            curBlockStr = ['That was Block. No. ', num2str(iTrl/blk_trials), ...
                '. \n' num2str(num_block-iTrl/blk_trials) ' blocks left. \n', ...
                '. \n' num2str((blkCorrect)) '% Correct. \n' ];
            cond = exp.getCondition;
            if withMRI
                v.dispMirrorText([curBlockStr infoText.blkText]);
            else
                v.dispText([curBlockStr infoText.blkText]);
            end
            WaitSecs(5);
        end
        
        %-------7. debugging needs 'stop' function------- %
        kb2.response(0)
        if kb2.wantStop %stop for debugging
            break;
        end
        
    end %end of trials
    
    
    %%%%%%%%%%%%%%show fixtation in the end %%%%%%%%%%%%%%
    v.dispFixation(20,2);
    WaitSecs(6);
    endtime=GetSecs;
    exp.u=[triggertime endtime]; %save trigger time
    exp.saveData;   %save data
    
    % check trigger
    while KbEventAvail(deviceIndex)
        key_ev = KbEventGet(deviceIndex);
        if (~isempty(key_ev) && key_ev.Pressed)
            trigger(count)=key_ev.Time;
            count=count+1;
            key_ev=[];
        end
        WaitSecs('Yieldsecs', 0.001);  % wait for 1 msec to prevent system overload
    end
    
    %%%%%%%%%%%%%%close the experiment%%%%%%%%%%%%%%%%%%%%
    
    % save trigger and close the queue
    trigger_name=[exp.sName '_trig.mat'];
    save([pwd '/trig/',trigger_name], 'trigger');
    KbQueueStop;
    KbQueueRelease;
    
    % close exp
    if withMRI
        v.dispMirrorText(infoText.thankyou);
    else
        v.dispText(infoText.thankyou);
    end
    
    WaitSecs(1);
    
    % close eyelink
    if withEyeLink
        Eyelink('CloseFile');
        Eyelink('ReceiveFile', [exp.sName '.edf']);
        Eyelink('Shutdown');
    end
    
    % close screen
    sca;
    if withEyeLink
        movefile([exp.sName, '.edf'], 'EDF');
    end
    
catch ME
    
    disp(ME.message);
    disp(ME.stack);
    for iTrl=1:length(ME.stack)
        disp(ME.stack(iTrl).name);
        disp(ME.stack(iTrl).line);
    end
    
    
end  % end of try
end % end of experiment

function infoText = init_text1(group)
% specify experimental text
if group==1
    infoText.instruction = ['Visual Search \n', ...
        'One bar will have a different orientation (tilted to right 12�) from the others, this is the target.', ...
        'The task is to find the target bar and judge whether the "dot" is on the top or the bottom.' ...
        'If the target has a "dot" on the TOP of the line, please press the LEFT button; \n', ...
        'If the target has a "dot" on the BOTTOM of the line, please press the RIGHT button.\n',...
        'Sometimes, there will also be an item with different color (red) or different oritation (tilted to left 45�). \n' ...
        'This is not the target, please ignore it. \n' ...
        'Please try to respond as quickly and accurately as possible.\n',...
        '\n\n Please be ready'];
else
    infoText.instruction = ['Visual Search \n', ...
        'One bar will have a different orientation (tilted to right 12�) from the others, this is the target.', ...
        'The task is to find the target bar and judge whether the "dot" is on the top or the bottom.' ...
        'If the target has a "dot" on the BOTTOM of the line, please press the LEFT button; \n', ...
        'If the target has a "dot" on the TOP of the line, please press the RIGHT button.\n',...
        'Sometimes, there will also be an item with different color (red) or different oritation (tilted to left 45�). \n' ...
        'This is not the target, please ignore it. \n' ...
        'Please try to respond as quickly and accurately as possible.\n',...
        '\n\n Please be ready'];
end

infoText.question = '?';
infoText.blkText = ['Please take a break!'];
infoText.thankyou = 'Thank you!';
end

function infoText = init_text2(group)
% specify experimental text
if group==1
    infoText.instruction = ['Visual Search \n', ...
        'One bar will have a different orientation (tilted to left 12�) from the others, this is the target.', ...
        'The task is to find the target bar and judge whether the "dot" is on the top or the bottom.' ...
        'If the target has a "dot" on the TOP of the line, please press the LEFT button; \n', ...
        'If the target has a "dot" on the BOTTOM of the line, please press the RIGHT button.\n',...
        'Sometimes, there will also be an item with different color (red) or different oritation (tilted to right 45�). \n' ...
        'This is not the target, please ignore it. \n' ...
        'Please try to respond as quickly and accurately as possible.\n',...
        '\n\n Please be ready'];
else
    infoText.instruction = ['Visual Search \n', ...
        'One bar will have a different orientation (tilted to left 12�) from the others, this is the target.', ...
        'The task is to find the target bar and judge whether the "dot" is on the top or the bottom.' ...
        'If the target has a "dot" on the BOTTOM of the line, please press the LEFT button; \n', ...
        'If the target has a "dot" on the TOP of the line, please press the RIGHT button.\n',...
        'Sometimes, there will also be an item with different color (red) or different oritation (tilted to right 45�). \n' ...
        'This is not the target, please ignore it. \n' ...
        'Please try to respond as quickly and accurately as possible.\n',...
        '\n\n Please be ready'];
end

infoText.question = '?';
infoText.blkText = ['Please take a break!'];
infoText.thankyou = 'Thank you!';
end



