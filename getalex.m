function lgraph=getalex(nout) 
net=alexnet;
inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
% inputSize = net(1).InputSize
% layersTransfer = net(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(128,'Name','fc128','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu_out')
    batchNormalizationLayer('Name','BN')
    fullyConnectedLayer(nout,'Name','fc2222','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax_out')
    classificationLayer('Name','classoutput')];

 lgraph = layerGraph(layers);
end
