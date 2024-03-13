%%run demofig1c first
%demofig1c
close all
savevid = true;
if savevid
    vw = vision.VideoFileWriter('VideoS1.mp4');
    vw.FileFormat = 'MPEG4';
    vw.FrameRate = 30;
end
numfollowpts = 4;

% Overwriting umix and dmix
umix = [0 7210 0]; %this is the data point for the second occurence of the stimulus (occuring at video frame 3210, 7210, and 11210)
dmix = [0 7710 0]; %500 frames = 250 ms stim end


%fig = figure('Position',[100 100 1280 768],'Color','k');
fig = figure('Position',[100 100 1280 768],'Color','k');
lread = VideoReader('SideCam_000000.avi');
%rread = VideoReader('RIGHTCAM_20190919_1623.avi');
%axl = axes('Position',[0 0.375 .5 0.625]);
axl = axes('Position',[0 0.375 1 0.625]); %changed to 1 to make the whole width of the box?
iml = imshow(zeros(480,640,'uint8'));
hold(axl,'on')
lmark = scatter(axl,nan(1,numfollowpts),nan(1,numfollowpts),linspace(1,150,numfollowpts),[102,194,165]./255,'Filled');
fpstext = text(axl,2,20,{'Recording FPS: 2000', ['Playback FPS: ' num2str(vw.FrameRate)]},'Color','k');
hold(axl,'off');
%axr = axes('Position',[.5 0.375 .5 .625]);
%imr = imshow(zeros(240,320,'uint8'));
%hold(axr,'on')
%rmark = scatter(axr,nan(1,numfollowpts),nan(1,numfollowpts),linspace(1,150,numfollowpts),[142,1,82]./255,'Filled');
% rtext = text(axr,0,220,{'Treated', 'Haltere'},'Color',[142,1,82]./255,'Fontsize',16);
%hold(axr,'off');
datax = axes('Position',[0 0 .5 .375]);
hold(datax,'on');
lline = plot(datax,haltangl,'Color',[102,194,165]./255,'LineWidth',2);
%rline = plot(datax,haltangr,'Color',[142,1,82]./255,'LineWidth',2);
%wline = plot(datax,wingang,'--','Color',[254,224,139]./255,'LineWidth',2);
scaleline = plot(nan(1,2),[-.06 -.06],'w','LineWidth',2);
hold(datax,'off');
datax.YLim = [-.15 1];
datax.XTick = [];
datax.YTick = [];
datax.Color = 'k';

ttext = text(datax,0,-.06,'5ms','Color','w','Fontsize',16);
%rtext = text(datax,0,-.06,'Treated Haltere','Color',[142,1,82]./255,'Fontsize',16);
ltext = text(datax,0,-.06,'Haltere','Color',[102,194,165]./255,'Fontsize',16);
%wtext = text(datax,0,-.06,'Wing','Color',[254,224,139]./255,'Fontsize',16);

dataxR = axes('Position',[.5 0 .5 .375]);
hold(dataxR,'on');
llineR = plot(dataxR,nan(size(haltangl)),'Color',[102,194,165]./255,'LineWidth',2);
%rlineR = plot(dataxR,nan(size(wingang)),'Color',[142,1,82]./255,'LineWidth',2);
%wlineR = plot(dataxR,nan(size(wingang)),'--','Color',1-[77,77,77]./255,'LineWidth',2);
scalebarR = plot(dataxR,[umix(2)-240 umix(2)-140],[-.06 -.06],'w','LineWidth',2);
scaletext = text(umix(2)-135,-.06,'50ms','Color','White','Fontsize',16);

magonline = plot(dataxR,[umix(2) umix(2)],[-.15 1],'--w');
magoffline = plot(dataxR,[dmix(2) dmix(2)],[-.15 1],'--w');
magonline.Visible = 'off';
magoffline.Visible = 'off';
magontext = text(umix(2)+10,-.06,'Light ON','Color','White','Fontsize',16);
magofftext = text(dmix(2)+10,-.06,'Light OFF','Color','White','Fontsize',16);
magontext.Visible = 'off';
magofftext.Visible = 'off';

hold(dataxR,'off');
dataxR.XTick = [];
dataxR.YTick = [];
dataxR.Color = 'k';
dataxR.XLim = [umix(2)-250 dmix(2)+249];
dataxR.YLim = [-.15 1];

lline.YData(1:umix(2)-249)=nan;
%rline.YData(1:umix(2)-249)=nan;
%wline.YData(1:umix(2)-249)=nan;



for i =umix(2)-250:dmix(2)+249
    imleft = rgb2gray(read(lread,i));
    %imright = rgb2gray(read(rread,i));
    iml.CData = imleft;
    lmark.XData = dlchaltlpts(i-(numfollowpts-1):i,1);
    lmark.YData = dlchaltlpts(i-(numfollowpts-1):i,2);
    %imr.CData = imright;
    %rmark.XData = dlchaltrpts(i-(numfollowpts-1):i,1);
    %rmark.YData = dlchaltrpts(i-(numfollowpts-1):i,2);
%     rmark.CData = repmat([142,1,82]./255,4,1).*linspace(0.1,1,4)';
    datax.XLim = [i-100 i];
    
    scaleline.XData = [i-99 i-89];    
    
    ttext.Position(1)=i-88;
    %rtext.Position(1)=i-65;
    ltext.Position(1)=i-40;
    %wtext.Position(1)=i-12;
    %rlineR.YData(i) = haltangr(i); 
    llineR.YData(i) = haltangl(i);
    
    if i == umix(2)
        magonline.Visible = 'on';
        magontext.Visible = 'on';
    end
    if i == dmix(2)
        magoffline.Visible = 'on';
        magofftext.Visible = 'on';
    end
    
    drawnow
    if savevid
        F = getframe(fig);
        I = frame2im(F);
        step(vw,I);
    end
end
if savevid;release(vw);end