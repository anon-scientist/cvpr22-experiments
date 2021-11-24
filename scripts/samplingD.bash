# MNIST pooling 3 GMMs
python3  -m DCGMM.utils.executeExp bash/DCGMM-D.bash --T1 1 4  --exp_id 5677 --epochs 50 \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L5_K 25 --L8_K 25 \
--sampling_layer 8 --L5_wait 0.1 --L8_wait 0.15
#./tmp.bash


python3  -m DCGMM.utils.executeExp bash/DCGMM-D.bash --T1 1 4 --T2 1 4  --exp_id 5677 --epochs 0.1 --saveAll True \
--load_task 1 --replay False --saveAll True --measuring_points 1 \
--L2_sampling_S 2 --L2_sampling_I -1 --L5_sampling_S 2 --L5_sampling_I -1 --L8_sampling_I -1 --L8_sampling_S -1 \
--L1_sharpening_iterations 500  --L1_sharpening_rate 0.1  --L3_sampling_mode dense --L6_sampling_mode dense \
--dataset_file MNIST --batch_size 100 \
--L2_K 25 --L5_K 25 --L8_K 25 \
--sampling_layer 8 --L5_wait 0.1 --L8_wait 0.15
./tmp.bash


