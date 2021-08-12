PYTHONPATH=.:.. python ./train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix \
	--direction BtoA --checkpoints_dir ./checkpoints_baseline --norm instance
# 150 * 400 => 60000
