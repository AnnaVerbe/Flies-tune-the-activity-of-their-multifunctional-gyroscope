close all
clear all

%dlchaltlfile = 'SideCam_000000DLC_resnet50_Halt59xS01Nov14shuffle1_100000.csv';
%dlchaltlpts = csvread(dlchaltlfile,3,1);
%dlchaltlpts = dlchaltlpts(:,1:2);
%dlchaltlpts(:,2) = dlchaltlpts(:,2);
dlchaltlfile = load('SideCam_000000_haltamp.mat'); %output from Amylyzer that has the corrected points
dlchaltlpts = dlchaltlfile.haltpos;
TF = find(dlchaltlpts(:,2)<120); %had to do this because Ampalyzer wasn't saving the corrections
dlchaltlpts(TF,2) = 200; %this needs to be deleted for any other files
hlrootx = dlchaltlfile.haltrpos(1,1); hlrooty = dlchaltlfile.haltrpos(1,2);
wlrootx =dlchaltlfile.wingrpos(1,1); wlrooty = dlchaltlfile.wingrpos(1,2);


%dlchaltrfile = 'RIGHTCAM_20190919_1623DeepCut_resnet50_PushPullFliesAug6shuffle1_1030000.csv';
%dlchaltrpts = csvread(dlchaltrfile,3,1);
%dlchaltrpts = dlchaltrpts(:,1:2);
%dlchaltrpts(:,2) = dlchaltrpts(:,2);

%dlcwingrfile = 'RIGHTCAM_20190919_1623DeepCut_resnet50_SideviewWingTrackerOct25shuffle1_1030000.csv';
%dlcwingrpts = csvread(dlcwingrfile,3,1);
%dlcwingrpts = dlcwingrpts(:,1:2);
%dlcwingrpts(:,2) = dlcwingrpts(:,2);

%hrrfile = 'RIGHTCAM_20190919_1623.haltroot';
%load(hrrfile,'-mat');
%hrrootx=pt(1);hrrooty=pt(2);    
%wrrfile = 'RIGHTCAM_20190919_1623.wingroot';
%load(wrrfile,'-mat');
%wrrootx=pt(1);wrrooty=pt(2);
%rbang = atan2d(wrrooty-hrrooty,wrrootx-hrrootx);

%hlrfile = 'SideCam_000000_haltamp.mat';
%load(hlrfile,'-mat');
%hlrootx=pt(1);hlrooty=pt(2);
%hlrootx = haltrpos(1,1); hlrooty = haltrpos(1,2);
%wlrfile = 'LEFTCAM_20190919_1623.wingroot';
%load(wlrfile,'-mat'); %<this was all done upon loading in the Amylyzer
%wlrootx=pt(1);wlrooty=pt(2);
%wlrootx = wingrpos(1,1); wlrooty = wingrpos(1,2);

lbang = atan2d(wlrooty-hlrooty,wlrootx-hlrootx);

haltangl = wrapTo360(rad2deg(unwrap(atan2(dlchaltlpts(:,2)-hlrooty,dlchaltlpts(:,1)-hlrootx)))-lbang);
haltangl = 360-haltangl;
[lhhi,lhlo]=envelope(haltangl,25,'Peak');
%[lhhiy,lhloy]=envelope(dlchaltlpts(:,2),25,'Peak');
haltangl = haltangl-min(haltangl);
haltangl = smooth(haltangl/max(haltangl));

%haltangr = wrapTo360(rad2deg(unwrap(atan2(dlchaltrpts(:,2)-hrrooty,dlchaltrpts(:,1)-hrrootx)))-rbang);
%[rhhi,rhlo]=envelope(haltangr,25,'Peak');
%haltangr = haltangr-min(haltangr);
%haltangr = haltangr/max(haltangr);

%wingang = wrapTo360(rad2deg(unwrap(atan2(dlcwingrpts(:,2)-wrrooty,dlcwingrpts(:,1)-wrrootx)))-rbang);
%[whi,wlo]=envelope(wingang,25,'Peak');
%wingang = wingang-min(wingang);
%wingang = wingang/max(wingang);

load('59xS01_231107_141944_f1_r1.fly2','-mat');
%trigix=find(rec.daq.data(:,3)>2,1);
trigix = 1;
magupix=find(diff(rec.daq.data(:,3))>3); %when light turns on
magdownix=find(diff(rec.daq.data(:,3))<-3); %when light turns off
daqts = rec.daq.tstamps;    
fs = 2000;
camts = linspace(0,length(haltangl)/fs,length(haltangl));
toffset = camts(end)-daqts(trigix);
camts = camts-toffset;
midix=mean([magupix magdownix],2);
midt = daqts(midix); 

[~,umix] = min(abs(daqts(magupix)'-camts'));
[~,dmix] = min(abs(daqts(magdownix)'-camts'));
magepoch = camts>inf;
for i = 1:length(umix)
    magepoch(umix(i):dmix(i))=1;
end

%%
figure('Position',[100 100 800 600]) %[X,Y position, width, height]
for i = 1:2
    subplot(2,1,i)
    hold on
    %plot(camts,wingang,'--','Color',[77,77,77]./255)
    plot(camts,haltangl,'LineWidth',1,'Color',[1,102,94]./255)
    %plot(camts,haltangr,'LineWidth',2,'Color',[140,81,10]./255)
%     plot(daqts([magupix magupix; magdownix magdownix]),[0 1],'r','LineWidth',4)
    xticks([]);
    yticks([]);
%     xlim([midt(3)-1.1+i-1 midt(3)-1.05+i-1])
end
plot([midt(3)-1.1+i-1 midt(3)-1.09+i-1],[.5 .5],'k')

%%
figure('Position',[100 100 800 200])
hold on
plot(camts,(lhhi-lhlo),'Color',[1,102,94]./255);
%plot(camts,(rhhi-rhlo),'Color',[140,81,10]./255);
plot(daqts([magupix magupix; magdownix magdownix]),ylim,'k--','LineWidth',2)
xlim([9,10-.1])
plot([9.05 9.1],[120 120],'k')
plot([9.05 9.05],[120 170],'k')
xticks([]);
yticks([]);

%%
tix = 3;
nmagix=find(camts>(daqts(magupix(tix))-.5) & camts<daqts(magupix(tix)));
nmagix=[nmagix find(camts>daqts(magdownix(tix)) & camts<(daqts(magdownix(tix))+.5))];
%xnmag = dlchaltrpts(nmagix,1);
xnmagL = dlchaltlpts(nmagix,1);
%ynmag = 240-dlchaltrpts(nmagix,2);
ynmagL = 240-dlchaltlpts(nmagix,2);

magix=find(camts>daqts(magupix(tix))+.1 & camts<daqts(magdownix(tix)));
%xmag = dlchaltrpts(magix,1);
xmagL = dlchaltlpts(magix,1);
%ymag = 240-dlchaltrpts(magix,2);
ymagL = 240-dlchaltlpts(magix,2);
figure
k= boundary(xnmagL,ynmagL);
plot(xnmagL(k),ynmagL(k),'k');hold on;
k= boundary(xmagL,ymagL);
plot(xmagL(k),ymagL(k),'r');
plot(hlrootx,hlrooty,'Xk')
axis square
%%
close all
%correct=pdist([hrrootx hrrooty;wrrootx wrrooty])/pdist([hlrootx hlrooty;wlrootx wlrooty]); %correct lcoords to rcoords
convert = 17.5689/152.4; %width of tungsten pin in right image
%fr=figure;
%hold on
%plot([repmat(hrlootx,size(xnmag)) xnmag]',[repmat(hrrooty,size(ynmag)) ynmag]','color',[.3 .3 .3]);
%h=plot(xnmag,ynmag,'k.');
% h.MarkerFaceColor = 'k';
%plot([repmat(hrrootx,size(xmag)) xmag]',[repmat(hrrooty,size(ymag)) ymag]','Color',[.9 .5 .5])
%h=plot(xmag,ymag,'r.');
%xlim([145 190])
%ylim([100 145]);
%xticks([]);
%yticks([]);
%plot([170 170],[130 130+(50*convert)],'k');
%plot([170 170+(50*convert)],[130 130],'k');
% h.MarkerFaceColor = 'r';
axis square
fr.Renderer='Painters';

fl=figure;
hold on
plot([repmat(hlrootx,size(xnmagL)) xnmagL]',[repmat(240-hlrooty,size(ynmagL)) ynmagL]','color',[.3 .3 .3]);
h=plot(xnmagL,ynmagL,'k.');
plot([repmat(hlrootx,size(xmagL)) xmagL]',[repmat(240-hlrooty,size(ymagL)) ymagL]','Color',[.9 .5 .5])
h=plot(xmagL,ymagL,'r.');
%xlim([195 240])
%ylim([125 180]);
xticks([]);
yticks([]);
axis square
fl.Renderer='Painters';
