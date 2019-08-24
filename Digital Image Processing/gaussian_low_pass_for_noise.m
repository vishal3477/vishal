figure;
i=imread('cameraman.tif');
subplot(2,3,1),imshow(i);
subplot(2,3,4),imhist(i);

noisy=imnoise(i,'poisson');
subplot(2,3,2),imshow(noisy);
subplot(2,3,5),imhist(noisy);
i1=fft2(noisy);
i2=fftshift(i1);

i3=log(1+abs(i2));
i4=mat2gray(i3);

[r,c]=size(i4);
filt=zeros(r,c);
d=15;
for i=1:r
    for j=1:c
        dist= sqrt((i-r/2)^2+(j-c/2)^2);
        filt(i,j)=exp(-dist*dist/(2*d*d));
    end
end

filtered_im=filt.*i2;
subplot(3,2,4),imshow(filtered_im);

i6=ifft2(filtered_im);
i7=mat2gray(abs(i6));
subplot(2,3,3),imshow(i7);
subplot(2,3,6),imhist(i7);