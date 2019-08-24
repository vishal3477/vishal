figure;
i=imread('cameraman.tif');
subplot(2,4,1),imshow(i);
subplot(2,4,5),imhist(i);

noisy=imnoise(i,'salt & pepper');
subplot(2,4,2),imshow(noisy)
subplot(2,4,6),imhist(noisy)

filtered_im=medfilt2(noisy);
subplot(2,4,3),imshow(filtered_im);
subplot(2,4,7),imhist(filtered_im);

%implementing median filter
[r,c]=size(noisy);
filtered_im1=zeros(r,c);
for i=2:r-1
    for j=2:c-1
        arr=[noisy(i-1,j-1),noisy(i-1,j),noisy(i-1,j+1),noisy(i,j-1),noisy(i,j),noisy(i,j+1),noisy(i+1,j-1),noisy(i+1,j),noisy(i+1,j+1)];
        med=median(median(arr));
        filtered_im1(i,j)=med;
    end
end
filtered_im1(1:r,c)=noisy(1:r,c);
filtered_im1(1:r,1)=noisy(1:r,1);
filtered_im1(1,1:c)=noisy(1,1:c);
filtered_im1(r,1:c)=noisy(r,1:c);
filtered_im2=mat2gray(filtered_im1);
subplot(2,4,4),imshow(filtered_im2);
subplot(2,4,8),imhist(filtered_im2);