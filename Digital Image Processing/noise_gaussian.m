im1=zeros(256,256);
im2=im1+20;

im2(30:226,30:226)=120;

for i=1:256
    for j=1:256
        x=128;
        y=128;
        dist=sqrt((i-x)^2+(j-y)^2);
        if dist<40
            im2(i,j)=220;
        end
    end
end

im3=im2/255;
figure;
subplot(2,3,1),imshow(im3);
subplot(2,3,4),imhist(im3);

noisy=imnoise(im3,'gaussian');
subplot(2,3,2),imshow(noisy);
subplot(2,3,5),imhist(noisy);

filtered_im=medfilt2(noisy);
subplot(2,3,3),imshow(filtered_im);
subplot(2,3,6),imhist(filtered_im);