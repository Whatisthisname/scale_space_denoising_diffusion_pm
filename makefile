make test:
	python3


make full:
	pip install -r requirements.txt
	#
	#
	# train the baseline model 1st time
	python3 train_ddpm.py --run_name "baseline" --img_size 28 --epochs 10  --markov_states 30 --noise_power 1.5
	python3 sample_DDPM.py --run_name "baseline" --size 3334 --compute_speed
	# train the baseline model 2nd time
	python3 train_ddpm.py --run_name "baseline" --img_size 28 --epochs 10  --markov_states 30 --noise_power 1.5
	python3 sample_DDPM.py --run_name "baseline" --size 3334 --stack_samples
	# train the baseline model 3rd time
	python3 train_ddpm.py --run_name "baseline" --img_size 28 --epochs 10  --markov_states 30 --noise_power 1.5
	python3 sample_DDPM.py --run_name "baseline" --size 3334 --stack_samples
	#
	#
	#
	#
	#
	# train the small model,1st half of cascaded, 1st time
	python3 train_ddpm.py --run_name "small" --img_size 14 --epochs 10  --markov_states 20 --noise_power 1.5
	python3 sample_DDPM.py --run_name "small" --size 3334 --rescale 28 -o "small upscaled" --compute_speed
	# train the small model,1st half of cascaded, 2nd time
	python3 train_ddpm.py --run_name "small" --img_size 14 --epochs 10  --markov_states 20 --noise_power 1.5
	python3 sample_DDPM.py --run_name "small" --size 3334 --rescale 28 -o "small upscaled" --stack_samples
	# train the small model,1st half of cascaded, 3rd time
	python3 train_ddpm.py --run_name "small" --img_size 14 --epochs 10  --markov_states 20 --noise_power 1.5
	python3 sample_DDPM.py --run_name "small" --size 3334 --rescale 28 -o "small upscaled" --stack_samples
	#
	#
	#
	#
	#
	# train the big model, 2nd half of cascaded, 1st time.
	python3 train_ddpm_big.py --run_name "big" --img_size 28 --epochs 10  --markov_states 10 --noise_power 1.5
	python3 sample_cascaded_DDPM.py --small_name "small" --big_name "big" --size 3334 --compute_speed
	# train the big model, 2nd half of cascaded, 2nd time.
	python3 train_ddpm_big.py --run_name "big" --img_size 28 --epochs 10  --markov_states 10 --noise_power 1.5
	python3 sample_cascaded_DDPM.py --small_name "small" --big_name "big" --size 3334 --stack_samples
	# train the big model, 2nd half of cascaded, 3rd time.
	python3 train_ddpm_big.py --run_name "big" --img_size 28 --epochs 10  --markov_states 10 --noise_power 1.5
	python3 sample_cascaded_DDPM.py --small_name "small" --big_name "big" --size 3334 --stack_samples
	#
	#
	#
	#
	#
	# compute CAS score
	#
	python3 compute_CAS.py --run_names baseline "small upscaled" small_to_big