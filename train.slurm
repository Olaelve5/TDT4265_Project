#!/bin/sh
#SBATCH --partition=GPUQ          
#SBATCH --gres=gpu:1              
#SBATCH --time=14:00:00           
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32000               
#SBATCH --job-name="rbk_train"    
#SBATCH --output=train_log.txt    


module load Python/3.10.8-GCCcore-12.2.0
source football_env/bin/activate

python train_model.py