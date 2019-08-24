clear all
close all

i=imread('rice.png');
[r,c]=size(i);
i2=zeros(r,c);
w=[0,-1,0;-1,4,-1;0,-1,0];
i1=im2double(i);
for j=2:r-1
    for k=2:c-1
        i2(j,k)=sum(sum(i1(j-1:j+1,k-1:k+1).*w));
    end
end
figure;
i3=i1+i2;
subplot(2,2,1),imshow(i);
subplot(2,2,2),imshow(i1);
subplot(2,2,3),imshow(i2);
subplot(2,2,4),imshow(i3);