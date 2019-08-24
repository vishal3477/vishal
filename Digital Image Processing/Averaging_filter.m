f=ones(3,3);
f1=f/9;
im=imread('cameraman.tif');
filteredim=imfilter(im,f1);
subplot(1,2,1),imshow(im);
subplot(1,2,2),imshow(filteredim);