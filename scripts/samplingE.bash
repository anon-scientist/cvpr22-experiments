# MNIST no pooling 3 GMMs
python3  -m DCGMM.utils.executeExp bash/DCGMM-E.bash --T1 1 4  --exp_id 5977 --epochs 50 \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 --L6_K 25 \
--sampling_layer 6 --L4_wait 0.1 --L6_wait 0.15
#./tmp.bash


python3  -m DCGMM.utils.executeExp bash/DCGMM-E.bash --T1 1 4 --T2 1 4  --exp_id 5977 --epochs 50 --saveAll True \
--load_task 1 --replay False --saveAll True\
--L2_sampling_S 2 --L2_sampling_I -1 --L4_sampling_S 2 --L4_sampling_I -1 --L6_sampling_I -1 --L6_sampling_S -1 \
--L1_sharpening_iterations 0  --L1_sharpening_rate 0.1  \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 --L6_K 25 \
--sampling_layer 6 --L3_wait 0.1 --L6_wait 0.15
./tmp.bash


