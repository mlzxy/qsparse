cd ./cyclegan_pix2pix
!PYTHONPATH=.:.. python ./train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
