%% Figure 6 Code

%% A: Graphic of organizational set-up

%% B: Haltere Line

%% C: Picture of the fly with the data plotted

%import the tracked data from DLTdv8a and the image of the fly

xy = udExport.data.xypts;
flyX = xy(:,1);
flyY = xy(:,2);

figure; imshow(cdata)
hold on
plot(flyX(1:25,:), flyY(1:25,:),"-o", 'Color', 'r')

%% D: Individual Traces

% Data tracked using DLC neural network trained on haltere position

rootdir = uigetdir();
recfiles = dir(fullfile(rootdir,'**/*haltamp.mat')); 
allangles = [];


for i = 1:length(recfiles)
    load(fullfile(recfiles(i).folder,recfiles(i).name),'-mat');
    haltxpos = haltpos(:,1);
    haltypos = haltpos(:,2);

    hrrootx = haltrpos(1,1);
    hrrooty = haltrpos(1,2);

    wrrooty = wingrpos(1,1);
    wrrootx = wingrpos(1,2);
    
    rbang = atan2d(wrrooty-hrrooty,wrrootx-hrrootx);
    
    haltangr = wrapTo360(rad2deg(unwrap(atan2(haltypos-hrrooty,haltxpos-hrrootx)))-rbang);
    
    [rhhi,rhlo]=envelope(haltangr,25,'Peak');
    haltangr = haltangr-min(haltangr);
    haltangr = haltangr/max(haltangr);
    plot(haltangr)


    allangles = [allangles angle];
 %   clear angle wingrpos haltpos
end

%plottingthefewtraces - 5 cycles each
figure; plot(0:56,haltangr(1006:1062)) %pre stim
hold on
plot(61:115,haltangr(2402:2456)) %during stim, near the end (stim 2000 - 2500)
plot(120:175, haltangr(8997:9052)) %post stim
box off

%% E: 3 traces overlaid from the same fly
% Data tracked using DLC neural network trained on haltere position

%converting to change in radians of amplitude
amplitude_angles = atan2(pos(:,1), pos(:,2));
amplitude = wrapTo2Pi(unwrap(amplitude_angles));
amplitude = amplitude - mean(amplitude);

amptrials = [];
for i = 1:size(amplitude,1)
    
    amptrials1 = amplitude(i, 751:2750); 
    amptrials2 = amplitude(i, 4751:6750);
    amptrials3 = amplitude(i, 8751:10750);


   amptrials = [amptrials; amptrials1; amptrials2; amptrials3];
end


timesingle = (linspace(0,2000,2000))/2000; %this goes from 0 to 1 second (250 pre, 750 post)
flynum = 11; %specify which fly to use

rowids = [flynum*3 - 2 : flynum*3];
ampinclude = amptrials(rowids,:);
figure; plot(timesingle, rad2deg(ampinclude))
xlabel('Time (s)')
ylabel('Change in Amplitude (Degrees)')
hold on
xline(0.25, '--r'); xline(0.5, '--r');

%% F: Summary Data with Confidence Interval & Control Comparison
% Data tracked using DLC neural network trained on haltere position
% Confidence Intervals done in Python with Dickerson Lab script


meanvec = mean(amptrials); %mean works across the rows
meancontrang = mean(controlamplitude); %mean across the control data of 59xwtEmpty

figure;plot(timesingle,rad2deg(meanvec)); box off %mean of the amplitude data
hold on
xline(0.25, '--r'); xline(0.5, '--r');
hold on
patch([timesingle fliplr(timesingle)], [rad2deg(confint(1,:)) rad2deg(fliplr(confint(2,:)))], 'r') %confidence interval patch for S01
plot(timesingle, meancontrang(1:2000)) %mean of the control data
patch([timesingle fliplr(timesingle)], [rad2deg(controlconfint(1,1:2000)) rad2deg(fliplr(controlconfint(2,1:2000)))], 'g') %confidence interval patch for control data
hold off