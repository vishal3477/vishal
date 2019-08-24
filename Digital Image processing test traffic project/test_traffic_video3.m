obj = VideoReader('traffic.avi');

darkCar = rgb2gray(read(obj,71));

% darkCarValue = 50;
% noDarkCar = imextendedmax(darkCar, darkCarValue);
% noDarkCar1 =im2bw(darkCar,0.55);

subplot(1,4,1),imshow(darkCar)
% subplot(1,3,3), imshow(noDarkCar1);

c=[1   1   60 120 160];
r=[160 100 25 25  120];
mask=roipoly(darkCar,c,r);
masked_image=zeros(120,160);
for i=1:120
    for j=1:160
        if mask(i,j)==1
            masked_image(i,j)=darkCar(i,j);
            else
            masked_image(i,j)=0;
        end
    end
end

    %im2=im2bw(im,0.7);
    im2=imextendedmax(masked_image,80);
    
    im3=imgaussfilt(mat2gray(im2),0.5);
    subplot(1,4,2),imshow(im3);
    
    mask=[-3,-10,-3;0,0,0;3,10,3];
    im4=imfilter(im3,mask);
    subplot(1,4,3),imshow(im4);
    
    %im5=mat2gray(im4);
    %im5=im2bw(im4);
%     sedisk=strel('disk',1);
%     im5= imclose(im4,sedisk);
    
    
    %im5=imextendedmax(im4,0.1);
    subplot(1,4,4),imshow(im4);
    
    im5 = imfill(im4, 'holes');
    figure,subplot(1,4,1),imshow(im6);
    im7 = imopen(im6, strel('rectangle', [3,3]));
    subplot(1,4,2),imshow(im7);
    im8 = imclose(im7, strel('rectangle', [7, 7]));
    subplot(1,4,3),imshow(im8);

    %im6=imopen(im5,strel('rectangle',[3,3]));
    im9=im2bw(im8);
    subplot(1,4,4),imshow(im9);
    
    H=vision.BlobAnalysis('BoundingBoxOutputPort', true,'AreaOutputPort', true, 'CentroidOutputPort', true,'MinimumBlobArea', 60);
    [area,centroid,bbox]=step(H,im9);
    final_image=insertShape(darkCar,'rectangle',bbox,'Color','yellow','LineWidth',2);
    figure,imshow(final_image);



    