i=imread('moon.tif');
subplot(3,2,1),imshow(i);
i1=fft2(i);
i2=fftshift(i1);

i3=log(1+abs(i2));
i4=mat2gray(i3);
subplot(3,2,2),imshow(i4);

[r,c]=size(i4);
filt=zeros(r,c);
d=100;
n=15;
for i=1:r
    for j=1:c
        dist= sqrt((i-r/2)^2+(j-c/2)^2);
        filt(i,j)=1/(1+(dist/d)^(2*n));
    end
end

subplot(3,2,3),imshow(filt);

filtered_im=filt.*i2;
subplot(3,2,4),imshow(filtered_im);

actual_filtered=filt.*i4;
subplot(3,2,5),imshow(actual_filtered);

i6=ifft2(filtered_im);
i7=mat2gray(abs(i6));
subplot(3,2,6),imshow(i7);