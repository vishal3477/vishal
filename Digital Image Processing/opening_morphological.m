im=imread('morphological.tif');
se=strel('disk',60);
im2=imerode(im,se);
im3=imdilate(im2,se);
im4=imopen(im,se);

subplot(1,3,1),imshow(im);
subplot(1,3,2),imshow(im3);
subplot(1,3,3),imshow(im4);