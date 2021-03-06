function drift = getDrift(scene_name)
    drift = 0.0;
    switch (scene_name)
        case 'F0'
           drift = 0.035;
        case 'F1'
           drift = 0.1;
        case 'F2'
           drift = 0.035;
        case 'F3'
           drift = 0.03;
        case 'F5'
           drift = 0.030;
        case 'F6'
           drift = 0.000;
        case 'F8'
           drift = 0.000;
    end
end