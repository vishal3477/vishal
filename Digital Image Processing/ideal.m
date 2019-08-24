clear
close all
clc

Kvco = 200e6;
Fr = 1/(50e-9);
set(0, 'DefaultTextFontSize', 16) ;
set(0, 'DefaultAxesFontSize', 16) ;
set(0, 'DefaultLineLineWidth', 2) ;
set(0, 'DefaultAxesXGrid', 'on') ;
set(0, 'DefaultAxesYGrid', 'on') ;


%%%%%% Settling Behaviour Extraction
data_1 = csvread('ds-spur-6p25.csv',1,0);
len=length(data_1);
num_extraction_cycles = 1024;
v_extract = data_1((len-num_extraction_cycles*2048+1):len,2);
t_extract = data_1((len-num_extraction_cycles*2048+1):len,1);
%%%%%% Start FFT computation
N_fft = 2048*64;
overlap_fact = N_fft/2;
fs = 20e6*2048;

[S_1,f_1]=psd(v_extract,N_fft,fs,N_fft,overlap_fact,'linear');

S_1 = 1.5*S_1/N_fft;
S_1(2:end) = S_1(2:end) .* ((Kvco ./ (2*f_1(2:end))).^2);
k=length(f_1);
f_1=f_1/Fr;

figure
semilogx(f_1(2:k),10*log10(abs(S_1(2:k))),'k','LineWidth',2)
hold on;
xlabel('Offset from carrier ($f/f_r$)','Interpreter','latex');
ylabel('Spur (in dBc)','Interpreter','latex');
grid on

%%%%%% End FFT Computation


%%%%%% Settling Behaviour Extraction
data_2 = csvread('ds-spur-12p5.csv',1,0);
len=length(data_2);
num_extraction_cycles = 1024;
v_extract = data_2((len-num_extraction_cycles*2048+1):len,2);
t_extract = data_2((len-num_extraction_cycles*2048+1):len,1);
%%%%%% Start FFT computation
N_fft = 2048*64;
overlap_fact = N_fft/2;
fs = 20e6*2048;

[S_2,f_2]=psd(v_extract,N_fft,fs,N_fft,overlap_fact,'linear');

S_2 = 1.5*S_2/N_fft;
S_2(2:end) = S_2(2:end) .* ((Kvco ./ (2*f_2(2:end))).^2);
k=length(f_2);
f_2=f_2/Fr;

semilogx(f_2(2:k),10*log10(abs(S_2(2:k))),'Color',[0.8,0,0],'LineWidth',2)


%%%%%% End FFT Computation

%%%%%% Settling Behaviour Extraction
data_3 = csvread('ds-spur-25n.csv',1,0);
len=length(data_3);
num_extraction_cycles = 1024;
v_extract = data_3((len-num_extraction_cycles*2048+1):len,2);
t_extract = data_3((len-num_extraction_cycles*2048+1):len,1);
%%%%%% Start FFT computation
N_fft = 2048*64;
overlap_fact = N_fft/2;
fs = 20e6*2048;

[S_3,f_3]=psd(v_extract,N_fft,fs,N_fft,overlap_fact,'linear');

S_3 = 1.5*S_3/N_fft;
S_3(2:end) = S_3(2:end) .* ((Kvco ./ (2*f_3(2:end))).^2);
k=length(f_3);
f_3=f_3/Fr;

semilogx(f_3(2:k),10*log10(abs(S_3(2:k))),'Color',[0.48,0.48,0.48],'LineWidth',2)
legend('Ideal Switch','Switch without Dummy','Switch with Dummy (10% Mis)')
axis([1/2,8,-200,-65])
%%%%%% End FFT Computation


%InSET
h1=figure(1);
h2=get(h1,'CurrentAxes');
h3=axes('pos',[0.5 0.8 0.1 0.25]);
set(h1,'CurrentAxes',h3)
hold on; grid on; 
semilogx(f_1(2:k),10*log10(abs(S_1(2:k))),'k','LineWidth',2)
semilogx(f_2(2:k),10*log10(abs(S_2(2:k))),'Color',[0.8,0,0],'LineWidth',2)
semilogx(f_3(2:k),10*log10(abs(S_3(2:k))),'Color',[0.48,0.48,0.48],'LineWidth',2)
axis([.95 1.05 -100 -68])


