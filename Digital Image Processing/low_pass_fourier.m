i=imread('cameraman.tif');
subplot(2,2,1),imshow(i);
i1=fft2(i);
i2=fftshift(i1);
i3=log(1+abs(i2));
i4=mat2gray(i3);
subplot(2,2,2),imshow(i4);

[r,c]=size(i4);
filt=zeros(r,c);
for i=1:r
    for j=1:c
        dist= sqrt((i-r/2)^2+(j-c/2)^2);
        if dist<90
            filt(i,j)=1;
        end
    end
end

filtered_im=filt.*i2;
subplot(2,2,3),imshow(filtered_im);

i6=ifft2(filtered_im);
i7=mat2gray(abs(i6));
subplot(2,2,4),imshow(i7);