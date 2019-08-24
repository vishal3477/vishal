i=imread('cameraman.tif');
i1=im2double(i);
i2=i1*255;
i3=zeros(256,256);
for x=1:1:256
    for y=1:1:256
        if i2(x,y)<70
            i3(x,y)=5*i2(x,y)/7;
        elseif i2(x,y)<150
            i3(x,y)=(13*i2(x,y)-560)/7;
        else
            i3(x,y)=
        end
    end
end
imshow(i3);