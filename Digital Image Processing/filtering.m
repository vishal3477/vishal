i=imread('cameraman.tif');
[r,c]=size(i);
i2=zeros(r,c);
w=[1,1,1;1,1,1;1,1,1]/9;
i1=im2double(i);
for j=2:r-1
    for k=2:c-1
        i2(j,k)=sum(sum(i1(j-1:j+1,k-1:k+1).*w));
    end
end
figure;
subplot(2,1,1),imshow(i);
subplot(2,1,2),imshow(i2);