i=imread('cameraman.tif');
noisy=imnoise(i,'gaussian');
subplot(2,2,1),imshow(noisy)

filter=ones(5,5)/25;
filtered_im=imfilter(noisy,ones);
subplot(2,2,2),imshow(filtered_im);
subplot(2,2,3),imhist(noisy);
subplot(2,2,4),imhist(filtered_im);