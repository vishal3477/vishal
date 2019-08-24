i=imread('cameraman.tif');
i1=fft2(i);
i2=fftshift(i1);
i3=log(1+abs(i2));
i4=mat2gray(i3);
imshow(i4);