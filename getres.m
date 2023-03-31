function lgraph=getres(nout)


net = resnet18;
 net.Layers
lgraph = layerGraph(net);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph)
%  inputSize = net.Layers(1).InputSize
% layersTransfer = net.Layers(1:end-1);
% numClasses = numel(categories(imdsTrain.Labels))
% 
lgraph = removeLayers(lgraph, {'ClassificationLayer_predictions','prob','fc1000'});
% 

nlayers = [
    fullyConnectedLayer(128,'Name','fc128','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu_out')
    batchNormalizationLayer('Name','BN')
    fullyConnectedLayer(nout,'Name','fc4','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax_out')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,nlayers);
lgraph = connectLayers(lgraph,'pool5','fc128');
end


%dropoutLayer('Name','Dropout')
%batchNormalizationLayer
%fullyConnectedLayer(512,'Name','fc512-2','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
% reluLayer('Name','relu_out-2')