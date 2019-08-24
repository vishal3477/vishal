im=imread('circles.png');
se=strel('disk',50);
im2=imerode(im,se);

im3=im-im2

subplot(1,3,1),imshow(im);
subplot(1,3,2),imshow(im2);
subplot(1,3,3),imshow(im3);