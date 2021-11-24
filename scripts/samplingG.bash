# MNIST pooling 4 GMMs
python3  -m DCGMM.utils.executeExp bash/DCGMM-G.bash --T1 1 4  --exp_id 4077 --epochs 50 --saveAll True \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L5_K 25 --L8_K 25 --L11_K 25 \
--sampling_layer 11 --L5_wait 0.1 --L8_wait 0.15 --L11_wait 0.2
#./tmp.bash


python3  -m DCGMM.utils.executeExp bash/DCGMM-G.bash --T1 1 4  --T2 1 4 --exp_id 4077 --epochs 50 --saveAll True \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L5_K 25 --L8_K 25 --L11_K 25 \
--sampling_layer 11 --L5_wait 0.1 --L8_wait 0.15 --L11_wait 0.2 \
--load_task 1 --replay False --saveAll True\
--L2_sampling_S 2 --L5_sampling_S 2  --L8_sampling_S 2 --L11_sampling_S 5 \
--L3_sampling_mode sparse --L6_sampling_mode dense --L9_sampling_mode dense \
--L1_sharpening_iterations 0  --L1_sharpening_rate 0.1  
./tmp.bash


