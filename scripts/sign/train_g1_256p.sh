python train.py --name sign_256p_g1 \
--dataroot datasets/sign --dataset_mode pose \
--input_nc 3 --ngf 64 --num_D 2 \
--resize_or_crop scaleHeight_and_scaledCrop --loadSize 288 --fineSize 256 \
--niter 5 --niter_decay 5 \
--n_frames_total 20 --max_frames_per_gpu 8 --max_t_step 4 \
--openpose_only
# python                           8717MiB