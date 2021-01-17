classdef CELoss < nnet.layer.RegressionLayer
    % Cross entropy loss layer.
    
    methods
        function layer = CELoss(name)
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Cross-entropy loss for binary classification';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the CE loss between
            % the predictions Y and the training targets T.

            % Calculate cross entropy
            crossEntropy = -(T.*log(Y) + (1-T).*log(1-Y));
    
            % Take mean over mini-batch.
            N = size(Y);
            N = N(end);
            loss = sum(crossEntropy,'all')/N;
        end
    end
end
