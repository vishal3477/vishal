%Author: Kartik Kulgod%
%Submitted


%Problem 1%
a=[1]; %filter characteristics%
b=[1,0,2,6,6,0,1];
n=0:50;
x=[n==0];%unit impulse signal%
h_part1=filter(b,a,x);%impulse response%
stem(h_part1),ylabel('impulse response h[n]'),xlabel('n'),title({'impulse response for';' y[n]=x[n]+2x[n-2]+6x[n-3]+6x[n-4]+x[n-6]'})
pause

[z,p,k]=tf2zpk(b,a);%to find zeros,poles and gain%
zplane(z,p),title('pole-zero plot of H(z)')
pause

disp('stable')
pause

inp_1=[1,-1,1,-1];%input signal%
out_1=conv(inp_1,h_part1);%output as a convolution of input and output signals%
subplot(1,2,1)
stem(n(1:4),inp_1),xlabel('n'),ylabel('x[n]'),title('input x[n] vs n'),axis([-1,5,-2,2])
subplot(1,2,2)
stem(n(1:10),out_1(1:10)),xlabel('n'),ylabel('y[n]'),title('output y[n] vs n')
pause

freqz(inp_1,[1]),title('input frequency response')%self-explanatory%
pause
freqz(out_1,[1]),title('output frequency response')%self-explanatory%
pause
freqz(h_part1,[1]),title('filter frequency response')%self-explanatory%
pause

%Problem 2%

a_2=[5,6,5];%filter characteristics%
b_2=[1];
h_part2=impz(b_2,a_2,21);%find 21 points of impulse response%
stem(n(1:21),h_part2),xlabel('n'),ylabel('h[n]'),title({'impulse response for';' y[n]=x[n]+2x[n-2]+6x[n-3]+6x[n-4]+x[n-6]'})
pause

[z_2,p_2,k_2]=tf2zpk(b_2,a_2);%find the zeros,poles and gain of H(z)%
zplane(z_2,p_2),title('Pole zero plot of H(z)')%pole zero plot of H(z)%
pause

disp('marginally stable')
pause

out_2=conv(inp_1,h_part2);
subplot(1,2,1)
stem(n(1:length(inp_1)),inp_1),xlabel('n'),ylabel('x[n]'),title('input as n')
subplot(1,2,2)
%size of output is N1+N2-1%
stem(n(1:(length(inp_1)+length(h_part2))-1),out_2),xlabel('n'),ylabel('y[n]'),title('output as n')
pause

freqz(inp_1,[1]),title('input frequency response')%self explanatory%
pause
freqz(out_2,[1]),title('output frequency response')%self explanatory%
pause
freqz(h_part2,[1]),title('filter frequency response')%self explanatory%
pause

%Problem 3%

x_3=[1,2,-6,6];%defining input signal%
stem(n(1:4),x_3)
pause

fft_x_4=fft(x_3,4);%4 point dft%
subplot(2,1,1)
stem(n(1:4),abs(fft_x_4)),xlabel('k'),ylabel('absolute X[k]'),title('Magnitude of 4-pint DFT of x[n]')
subplot(2,1,2)
stem(n(1:4),angle(fft_x_4)),xlabel('k'),ylabel('Phase X[k]'),title('Phase of 4-point DFT of x[n]')
pause

fft_x_8=fft(x_3,8);%8 point dft%
subplot(2,1,1)
stem(n(1:8),abs(fft_x_8)),xlabel('k'),ylabel('absolute X[k]'),title('Magnitude of 4-pint DFT of x[n]')
subplot(2,1,2)
stem(n(1:8),angle(fft_x_8)),xlabel('k'),ylabel('angle X[k]'),title('Phase of 4-pint DFT of x[n]')
pause




