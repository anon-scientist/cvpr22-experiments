# MNIST 2 Layer no pooling
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 1 4  --exp_id 2415 --epochs 50 \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 \
--sampling_layer 4 --L4_wait 0.1
./tmp.bash

python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 1 4  --T2 1 4 --exp_id 2415 --epochs 0.5 \
--load_task 1 \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 2 \
--L1_target_layer 4 --L1_reconstruction_weight 0.1 --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 \
--sampling_layer 4 --L4_wait 0.1
./tmp.bash

