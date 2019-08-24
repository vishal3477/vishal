i=imread('cameraman.tif');
figure;
subplot(2,3,1),imshow(i);
subplot(2,3,4),imhist(i);

noisy=imnoise(i,'gaussian');
subplot(2,3,2),imshow(noisy);
subplot(2,3,5),imhist(noisy);

[r,c]=size(noisy);
filtered_im=zeros(r,c);
for i=2:r-1
    for j=2:c-1
        arr=[noisy(i-1,j-1),noisy(i-1,j),noisy(i,j-1),noisy(i,j),noisy(i+1,j),noisy(i,j+1),noisy(i+1,j+1),noisy(i-1,j+1),noisy(i+1,j-1)];
        filtered_im(i,j)=max(max(arr));
    end
end
filtered_im(1:r,1)=noisy(1:r,1);
filtered_im(1:r,c)=noisy(1:r,c);
filtered_im(1,1:c)=noisy(1,1:c);
filtered_im(r,1:c)=noisy(r,1:c);

filtered_im1=mat2gray(filtered_im);

subplot(2,3,3),imshow(filtered_im1);
subplot(2,3,6),imhist(filtered_im1);