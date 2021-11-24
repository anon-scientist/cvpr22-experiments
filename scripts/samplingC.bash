# MNIST 2 Layer pooling
python3  -m DCGMM.utils.executeExp bash/DCGMM-C.bash --T1 1 4  --exp_id 9277 --epochs 50 \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L5_K 25  \
--sampling_layer 5 --L5_wait 0.1
#./tmp.bash


python3  -m DCGMM.utils.executeExp bash/DCGMM-C.bash --T1 1 4 --T2 1 4  --exp_id 9277 --epochs 50 --saveAll True \
--load_task 1 --replay False --saveAll True\
--L2_sampling_S 5 --L2_sampling_I -1 --L5_sampling_S 5 --L5_sampling_I -1  \
--L3_sampling_mode sparse \
--L1_sharpening_iterations 0  --L1_sharpening_rate 0.1   \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L5_K 25  \
--sampling_layer 5 --L5_wait 0.1
./tmp.bash


