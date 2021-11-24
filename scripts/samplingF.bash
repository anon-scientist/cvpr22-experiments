# MNIST no pooling 4 GMMs
python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 1 4  --exp_id 5077 --epochs 50 \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 --L6_K 25 --L8_K 25 \
--sampling_layer 8 --L4_wait 0.1 --L6_wait 0.15 --L8_wait 0.2
#./tmp.bash


python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 1 4 --T2 1 4  --exp_id 5077 --epochs 50 --saveAll True \
--load_task 1 --replay False --saveAll True\
--L2_sampling_S 2 --L2_sampling_I -1 --L4_sampling_S 2 --L4_sampling_I -1 --L6_sampling_I -1 --L6_sampling_S 2 --L8_sampling_S 5 \
--L1_sharpening_iterations 0  --L1_sharpening_rate 0.1  \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L4_K 25 --L6_K 25 --L8_K 25 \
--sampling_layer 8 --L4_wait 0.1 --L6_wait 0.15 --L8_wait 0.2
./tmp.bash


