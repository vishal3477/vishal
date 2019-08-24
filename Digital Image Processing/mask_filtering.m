function [filteredim]= mask_filtering(im,filt)
[l,m]=size(im);
[l1,m1]=size(filt);
filteredim=zeros(l,m);
imdouble=im2double(im);
for j=(l1+1)/2:l-(l1+1)/2
    for k=(m1+1)/2:m-(m1+1)/2
        x=(l1-1)/2;
        y=(m1-1)/2;
        neighborhood=imdouble(j-x:j+x,k-y:k+y);
        mult=neighborhood.*filt;
        S=sum(sum(mult));
        filteredim(j,k)=S;
    end
filteredim(1,:)=im(1,:);
filteredim(l,:)=im(l,:);
filteredim(:,m)=im(:,m);
filteredim(:,1)=im(:,1);
end
end