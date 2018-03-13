index = 1;
for i = 1:250
    if test_labels(i)== 2
        figure(index);
        index = index + 1;
        imshow(reshape(double(test_images(i ,:)),[28 28]));
    end;
end;