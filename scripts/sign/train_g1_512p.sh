python train.py --name sign_512p_g1 \
--dataroot datasets/sign --dataset_mode pose \
--input_nc 3 --n_scales_spatial 2 --ngf 64 --num_D 3 \
--resize_or_crop scaleHeight_and_scaledCrop --loadSize 576 --fineSize 512 \
--n_frames_total 20 --max_frames_per_gpu 4 --max_t_step 4 --add_face_disc \
--niter_fix_global 3 --niter 5 --niter_decay 5 \
--lr 0.0001 --load_pretrain checkpoints/sign_256p_g1 \
--openpose_only