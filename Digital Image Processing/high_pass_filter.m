im=imread('cameraman.tif');
subplot(2,3,1),imshow(im);
i1=fft2(im);
i2=fftshift(i1);

i3=log(1+abs(i2));
i4=mat2gray(i3);
subplot(2,3,2),imshow(i4);

[r,c]=size(i2);
highpass=zeros(r,c);
d0=50;
n=3;
for i=1:r
    for j=1:c
        dist=sqrt((i-r/2)^2+(j-c/2)^2);
        highpass(i,j)=1/(1+((d0/dist)^2)^n);
    end
end
subplot(2,3,3),imshow(highpass);

filteredim=i4.*highpass;
subplot(2,3,4),imshow(filteredim);

filterim=i2.*highpass;
subplot(2,3,5),imshow(filterim);

i5=ifft2(filterim);
finalim=mat2gray(abs(i5));
subplot(2,3,6),imshow(finalim);