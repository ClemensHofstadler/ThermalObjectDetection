classdef upsampleLayer < nnet.layer.Layer
    properties
       % which dimension should be split
        UpSampleFactor
    
    end
    
    properties (Learnable)
       % no learnable parameters 
    end
    
    methods
        function layer = upsampleLayer(factor, name)
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "upSamplingLayer with factor " + factor;
            
            % Stride by which layer is upsampled.
            layer.UpSampleFactor = factor;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = repelem(X,layer.UpSampleFactor,layer.UpSampleFactor);
        end
    end
end

