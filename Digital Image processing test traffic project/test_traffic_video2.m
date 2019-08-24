clear all
clc

%read video
traffic_video = VideoReader('traffic.avi');
nframes=traffic_video.NumberOfFrames;

%extracting frame1
frame1=read(traffic_video,1);

%creating empty video
final_video=zeros([size(frame1,1) size(frame1,2) 3 nframes],class(frame1));

for k=1:nframes
    %extracting kth frame of video
    current_frame=read(traffic_video,k);
    im=rgb2gray(current_frame);
    
    %extracting region of interest
    c=[1   1   60 120 160];
    r=[160 100 25 25  120];
    mask=roipoly(im,c,r);
    masked_image=zeros(120,160);
    for i=1:120
        for j=1:160
            if mask(i,j)==1
            masked_image(i,j)=im(i,j);
            else
            masked_image(i,j)=0;
            end
        end
    end
    
    im2=imextendedmax(masked_image,80);
    
    %gaussian filter
    im3=imgaussfilt(mat2gray(im2),0.5);
    
    %sobel filter
    mask=[-3,-10,-3;0,0,0;3,10,3];
    im4=imfilter(im3,mask);
    
    %removing small blobs 
    im5 = imfill(im4, 'holes');
    im6 = imopen(im5, strel('rectangle', [3,3]));
    im7 = imclose(im6, strel('rectangle', [7,7]));
    
    %converting image to binary
    im8=im2bw(im7);
    
    %using blob functio to get bounding box of blob
    H=vision.BlobAnalysis('BoundingBoxOutputPort', true,'AreaOutputPort', true, 'CentroidOutputPort', true,'MinimumBlobArea', 60);
    [area,centroid,bbox]=step(H,im8);
    
    %creating a rectangle on bounding boxes
    final_image=insertShape(current_frame,'rectangle',bbox,'Color','yellow','LineWidth',2);
    
    %storing kth frame with rectangle in final video
    final_video(:,:,:,k)=final_image;
    
end
frame_rate=traffic_video.FrameRate;
implay(final_video,frame_rate);


