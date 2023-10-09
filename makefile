make fake:
	python synthesize_dataset.py --run_name $(run) --size 10000

make train:
	python theo_train_ddpm.py --run_name $(run) --ckpt

make eval:
	python theo_eval.py --run_name $(run)