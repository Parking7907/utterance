CUDA_VISIBLE_DEVICES=1 python ./trainSpeakerNet.py --eval --model ResNetSE34F1 --log_input True --trainfunc aamsoftmax --margin 0.3 --save_path exps/test --eval_frames 400 --initial_model ./model000000240.model