figure;
im=imread('trimodal.tif');
subplot(2,2,1),imshow(im);
[im1,th1]=edge(im,'Canny',100/255);
[im2,th2]=edge(im,'Prewitt',100/255);
[im3,th3]=edge(im,'Sobel',100/255);
subplot(2,2,2),imshow(im1);
subplot(2,2,3),imshow(im2);
subplot(2,2,4),imshow(im3);