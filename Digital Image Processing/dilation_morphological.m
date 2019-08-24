im=imread('circles.png');
se=strel('disk',5);
im2=imdilate(im,se);

subplot(1,2,1),imshow(im);
subplot(1,2,2),imshow(im2);