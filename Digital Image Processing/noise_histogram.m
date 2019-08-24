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
subplot(2,2,1),imshow(im3);
subplot(2,2,3),imhist(im3);

noisy=imnoise(im3,'salt & pepper');
subplot(2,2,2),imshow(noisy);
subplot(2,2,4),imhist(noisy);