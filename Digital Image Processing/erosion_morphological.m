im=imread('circles.png');
se=strel('disk',10);
im2=imerode(im,se);

subplot(1,2,1),imshow(im);
subplot(1,2,2),imshow(im2);