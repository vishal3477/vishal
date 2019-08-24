clear all

N=400;
cyc=4;
x=(1:N)*cyc/N;
Isin(1,:)=sin(2*pi*x);
stem(1:400,x);

I_8=im2uint8(Isin);
stem(1:400,I_8)

for k=1:N
    i(k,:)=I_8;
end
figure;
imshow(i);