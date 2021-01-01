classdef splitInputLayer < nnet.layer.Layer   
    properties
       % no class variables
    end
    
    properties (Learnable)
        % no learnable parameters
    end
    
    methods
        function layer = splitInputLayer(name)
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Splits the input at the color-dimension.";
            
            layer.NumOutputs = 2;
            
        end
        
        function [Z1,Z2] = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the results Z1, Z2.
  
            % X = (w,h,2,batchSize,seq_length)
            Z1 = X(:,:,1,:,:);
            Z2 = X(1:3,1:4,2,:,:);
        end
    end
end

