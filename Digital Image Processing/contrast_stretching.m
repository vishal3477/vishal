i=imread('einstein_orig.tif');
i1=im2double(i);
i2=i1*255;

[r,c]=size(i);
i3=zeros(r,c);
for x=1:1:r
    for y=1:1:c
        if i2(x,y)<70
            i3(x,y)=5*i2(x,y)/7;
        elseif i2(x,y)<150
            i3(x,y)=(13*i2(x,y)-560)/7;
        else
            i3(x,y)=(5*i2(x,y)+510)/7;
        end
    end
end
i4=im2uint8(i3/255);
figure;
subplot(1,2,1),imshow(i);
subplot(1,2,2),imshow(i4);