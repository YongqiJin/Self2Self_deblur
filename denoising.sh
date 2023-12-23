python denoising.py \
    --path './testsets/barbara/' \
    --bs 1 \
    --sigma 25 \
    --iteration 150000 \
    --lr 1e-4 \
    --test_frequency 1000 \
    --num_prediction 100 \
    --log_path './logs/denoising/log_barbara.txt' \
    --device 'cuda:0'