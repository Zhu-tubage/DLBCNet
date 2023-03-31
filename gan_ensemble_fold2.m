clear;
% imds = imageDatastore('dataset', ...
%        'I·    ncludeSubfolders', true, ...
%        'LabelSource', 'foldernames');
% [imd1 imd2 imd3 imd4 imd5] = splitEachLabel(imds,0.2,0.2,0.2,0.2,'randomize');
% % load dense_imd;
% partStores{1} = imd1.Files ;
% partStores{2} = imd2.Files ;
% partStores{3} = imd3.Files ;
% partStores{4} = imd4.Files ;
% partStores{5} = imd5.Files ;

imdsTrain = imageDatastore('v1ganTRAIN', ...
       'IncludeSubfolders', true, ...
       'LabelSource', 'foldernames');
imdsTest = imageDatastore('v1TEST', ...
       'IncludeSubfolders', true, ...
       'LabelSource', 'foldernames');

layers=getvgg(4);
layers.Layers
inputSize = layers.Layers(1).InputSize;


% layers=getres(2);
% layers.Layers
% inputSize = layers.Layers(1).InputSize;

k = 5;
idx = crossvalind('Kfold', k, k);
% activation_layer = 'fc128';
activation_layer = 'fc128';
N = 400;

% start_all = tic
for i = 1:k
%     i
%     test_idx = (idx == i);
%     train_idx = ~test_idx;
%     imdsTest = imageDatastore(partStores{test_idx}, 'IncludeSubfolders',true,'LabelSource', 'foldernames');
%     imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'LabelSource', 'foldernames');
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb');
  
    
    options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 1, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 300, ...
    'ValidationPatience', Inf, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train = trainNetwork(augimdsTrain,layers,options);
% start_backbone = tic
TrainedNet = trainNetwork(augimdsTrain,layers,options);
           [YPred,scores] = classify(TrainedNet,augimdsValidation); 
      cfmres(:,:,i) = confusionmat(imdsTest.Labels, YPred)
% time_backbone = toc(start_backbone)

      trainy = imdsTrain.Labels;
      testy = imdsTest.Labels;
trainx = activations(TrainedNet,augimdsTrain,activation_layer,'OutputAs','rows');
testx = activations(TrainedNet, augimdsValidation,activation_layer, 'OutputAs','rows');
      [cfmelm(:,:,i),cfmsnn(:,:,i),cfmrvfl(:,:,i),cfmen(:,:,i),ta,TTest,TestScores]=entest(N,trainx,trainy,testx,testy);

% start_drvfl = tic
% net= dRVFLtrain(trainx,single(trainy),[2000 2000 2000 2000]);
% % net= dRVFLtrain(trainx, trainy, [5,5,5]);
% out=dRVFLtest(testx,net);
% 
% % cfmdrvfl(:,:,i) = confusionmat(testy, out);
% cfmdrvfl(:,:,i) = confusionmat(single(testy), single(out));
% 
% cfmdrvfl
%       [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
%            precision_net(i),F1_net(i)]=getindexes(cfmdrvfl(:,:,i));
% time_drvfl = toc(start_drvfl); 
% 
% cfmrvfl
%       [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
%            precision_net(i),F1_net(i)]=getindexes(cfmrvfl(:,:,i));
% cfmsnn
%       [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
%            precision_net(i),F1_net(i)]=getindexes(cfmsnn(:,:,i));
% cfmelm
%       [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
%            precision_net(i),F1_net(i)]=getindexes(cfmelm(:,:,i));
% cfmen
%       [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
%            precision_net(i),F1_net(i)]=getindexes(cfmen(:,:,i));
% [YPred,scores] = classify(TrainedNet,augimdsValidation);

YValidation = imdsTest.Labels;
accuracy(i) = mean(YPred == YValidation)
confMat(:,:,i) = confusionmat(YValidation, YPred)
confMat
      [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
           precision_net(i),F1_net(i)]=getindexes(confMat(:,:,i));
end
% toc
% time1 = toc(start_all)


% m_acc=mean(accuracy_net)
% m_sen=mean(sensitivity_net)
% m_spe=mean(specificity_net)
% m_pre=mean(precision_net)
% m_F1=mean(F1_net)
