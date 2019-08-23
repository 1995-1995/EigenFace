%When running this file please add 10 face pic of one person in NewTrainingSet directory and one
%pic of the same person in MyOwnFace directory

TopK = 15;            %%Set a topK value
%% This is the dataset with 135 given image
A = imread('Yale-FaceA/trainingset/subject01.centerlight.png');  %read one of image in training set in order to get image size
[m,n] = size(A);
%%Read images to form dataset
path_list = dir('Yale-FaceA/trainingset/*.png');
NoOfImg = length(path_list);
TrainingSet = zeros(m*n,NoOfImg);                         %Form a [m*n,NofImg] size matrix to st
for i=1:NoOfImg
    name = path_list(i).name;                                   %Name of pic
    I = imread(strcat('Yale-FaceA/trainingset/',name));
    I = double(I);
    I = reshape(I,m*n,1);                                         %transfer face image to a column vector
    TrainingSet(:,i) = I;
end
%Count means and difference 
Xmean = mean(TrainingSet,2);
DeltaTrainingSet = TrainingSet-Xmean;
%  Show average face
T=reshape(Xmean,m,n);
figure;
imshow(uint8(T));
%compute eigen values
Covariance = DeltaTrainingSet'*DeltaTrainingSet/NoOfImg;      %% eigen vector of A*A' can be computed by eigen vector of A'*A
[eigenVector,eigenValue] = eig(Covariance);
VectorizedValue = sum(eigenValue,2);                       %C is a symmetric vector, it can have a linear combination like C(1,1)+C(2,2)....+C(N,N)
[SortedValue, index] = sort(VectorizedValue,'descend');           %Sort computed vector in a descend order and record its index
totalEigen = sum(SortedValue);                              %compute the total value of eigen value
StoreEigenVector = [];
%Find k needed eigens
for i = 1:NoOfImg
    StoreEigenVector = [StoreEigenVector,eigenVector(:,index(i))];    %%Recored the vector that counted in loop
    if (i>=TopK )                                                     %When i reacheas TopK, break
        break;
    end
end
%Compute eigen face
EigenFace = DeltaTrainingSet*StoreEigenVector;
[m1,n1] = size(EigenFace);
% Display EigenFace 
figure;
for i = 1:n1
subplot(3,5,i)
EigenFace1 = reshape(EigenFace(:,i),m,n);
imshow(mat2gray(EigenFace1));
end
% Recognize face and show top 3 in training set
path_list2 = dir('Yale-FaceA/testset/*.png');
[m1,n1] = size(EigenFace);
for i=1:10
    name_ = path_list2(i).name;           %Name of pic
    I = imread(strcat('../Task3/Yale-FaceA/testset/',name_));
    I = double(I);
    Test = reshape(I,m*n,1);
    Test_Y = EigenFace'*(Test - Xmean);                              % Reflect the test sample to a new place
    TrainSet_Y =EigenFace'*DeltaTrainingSet;      % compute training set by EigenFace Value
    distance = zeros(n1,NoOfImg);                                      % make a container to store sdistance;
    for  j = 1:135
       distance(:,j)= (TrainSet_Y(:,j) - Test_Y).^2;
    end
    distance = sum(distance);
    [sorteddistance,index] = sort(distance,'ascend');
    numberOfCorrect = 0;
     figure;
     subplot(2,2,1);
     imshow(mat2gray(I));
     title('original')
    for pic = 1:3
        index_=index(pic);
        selected = TrainingSet(:,index_);
        shapeBack = reshape(selected,m,n);     
        subplot(2,2,pic+1);
        imshow(mat2gray(shapeBack));
        title(num2str(pic));
    end
end

%%
%% Display with own face
% This is the dataset include my own imgages
A = imread('Yale-FaceA/trainingset/subject01.centerlight.png');  %read one of image in training set in order to get image size
[m,n,l] = size(A);
%%Read images to form dataset
path_list = dir('Yale-FaceA/NewTrainingSet/*.png');
NoOfImg = length(path_list);
TrainingSet = zeros(m*n,NoOfImg);                         %Form a [m*n,NofImg] size matrix to st
for i=1:NoOfImg
    name = path_list(i).name;           %Name of pic
    I = imread(strcat('Yale-FaceA/NewTrainingSet/',name));
    I = double(I);
    I = reshape(I,m*n,1);
    TrainingSet(:,i) = I;
end
%%Count means and difference 
Xmean = mean(TrainingSet,2);
DeltaTrainingSet = TrainingSet-Xmean;
%%compute eigen values
Covariance = DeltaTrainingSet'*DeltaTrainingSet/NoOfImg;      %% A*A' has the same eigen values as A'*A ; A*A' will be pretty large
[eigenVector,eigenValue] = eig(Covariance);
VectorizedValue = sum(eigenValue,2);                       %C is a symmetric vector, it can have a linear combination like C(1,1)+C(2,2)....+C(N,N)
[SortedValue, index] = sort(VectorizedValue,'descend');           %Sort computed vector in a descend order and record its index
percentage = 0;                                                    %record the percentage that can represent the face
totalEigen = sum(SortedValue);                              %compute the total value of eigen value
StoreEigenVector = [];
%Find needed eigens
for i = 1:NoOfImg
    percentage = percentage+SortedValue(i)/totalEigen;
    StoreEigenVector = [StoreEigenVector,eigenVector(:,index(i))];    %%Recored the vector that counted in loop
    if (i>=TopK )                                         % When the recorded values can represent 99% of the total eigen value,break
        break;
    end
end
%%Compute eigen face
EigenFace = DeltaTrainingSet*StoreEigenVector;
[m1,n1] = size(EigenFace);
%% Recognize face and show top 3 in training set
path_list2 = dir('Yale-FaceA/MyOwnFace/*.png');
NoOfImg2 = length(path_list2);
[m1,n1] = size(EigenFace);
for i=1:length(path_list2)
    name_ = path_list2(i).name;                              %Name of pic
    I = imread(strcat('Yale-FaceA/MyOwnFace/',name_));
    I = double(I);
    Test = reshape(I,m*n,1);
    Test_Y = EigenFace'*(Test - Xmean);                 % projection the test sample to a new face place defined by eigenfaces
    TrainSet_Y =EigenFace'*DeltaTrainingSet;          % compute training set by EigenFace Value
    distance = zeros(n1,NoOfImg);                           % make a container to store sdistance;
    for  j = 1:NoOfImg
       distance(:,j)= (TrainSet_Y(:,j) - Test_Y).^2;      %Compute square Euclidean distanc
    end
    distance = sum(distance);
    [sorteddistance,index] = sort(distance,'ascend');  %Sort distance in ascending order because we need to find nearest
    numberOfCorrect = 0;
     figure;
     subplot(2,2,1);
     imshow(mat2gray(I));
     title('original')
    for pic = 1:3
        index_=index(pic);                                        %Display top 3 nearest
        selected = TrainingSet(:,index_);
        shapeBack = reshape(selected,m,n);     
        subplot(2,2,pic+1);
        imshow(mat2gray(shapeBack));
        title(num2str(pic));
    end
end

