 #!/bin/bash

# baseline
#./run_smooth.sh 0 none &

# window with different alphas
./run_smooth.sh 1 window 0.1 &
./run_smooth.sh 2 window 0.3 &
./run_smooth.sh 3 window 0.5 &

# laplacian with different alphas
./run_smooth.sh 4 laplacian 0.1 &
./run_smooth.sh 5 laplacian 0.3 &
./run_smooth.sh 6 laplacian 0.5 &

# ema with different rhos
./run_smooth.sh 7 ema 0.5 0.3 &

wait
echo "All runs complete"