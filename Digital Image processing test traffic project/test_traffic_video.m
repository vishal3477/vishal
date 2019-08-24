obj = VideoReader('traffic.avi');

darkCar = rgb2gray(read(obj,71));

darkCarValue = 50;
noDarkCar = imextendedmax(darkCar, darkCarValue);
noDarkCar1 =im2bw(darkCar,0.55);

figure,subplot(1,3,1),imshow(darkCar)
subplot(1,3,2), imshow(noDarkCar);
subplot(1,3,3), imshow(noDarkCar1);