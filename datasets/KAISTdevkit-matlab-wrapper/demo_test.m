function demo_test()

%% specify path of detections
dtDir = 'C:\Users\batuy\Desktop\MS-CornerNet\datasets\KAISTdevkit-matlab-wrapper\detections\msds_sanitized\det';

%% specify path of groundtruth annotaions
gtDir = 'C:\Users\batuy\Desktop\kaist\annotations\test_improved';

%% specify path of saved data
savePath = 'C:\Users\batuy\Desktop\saves\test1\results';

%% evaluate detection results
kaist_eval_full(dtDir, gtDir, false, true, savePath);

end
