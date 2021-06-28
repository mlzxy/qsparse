cd ./cyclegan_pix2pix
!PYTHONPATH=.:.. python ./train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix \ 
--direction BtoA --checkpoints_dir ./checkpoints_sparse \
	--sparse "sparsity=0.5,start_steps=60000,prune_freq=4000,n_prunes=4"

# 150 * 400 => 60000
