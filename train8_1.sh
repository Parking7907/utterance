CUDA_VISIBLE_DEVICES=1 python ./trainSpeakerNet.py --model ResNetSE34FF2 --log_input True --encoder_type SAP --trainfunc amsoftmax --save_path exps/exp_F8_1 --nClasses 5994 --batch_size 200 --scale 30 --margin 0.3