make test:
	python3


make full:
	pip install -r requirements.txt
	#
	# train the baseline model
	#
	python3 train_ddpm.py --run_name "baseline" --img_size 28 --epochs 2  --early_stop 2
	#
	# train the small model
	#
	python3 train_ddpm.py --run_name "small" --img_size 14 --epochs 2  --early_stop 2
	#
	# train the upscaling model
	#
	python3 train_ddpm_big.py --run_name "big" --img_size 28 --epochs 2 --early_stop 2
	#
	# sample from the cascaded model for CAS
	#
	python3 sample_cascaded_DDPM.py --small_name "small" --big_name "big" --size 100
	#
	# sample from the baseline for CAS
	#
	python3 sample_DDPM.py --run_name "baseline" --size 100
	#
	# sample from the upscaled small model for CAS
	#
	python3 sample_DDPM.py --run_name "small" --size 100 --rescale 28 -o "small upscaled"
	#
	# compute CAS score
	#
	python3 compute_CAS.py --run_names baseline "small upscaled" small_to_big