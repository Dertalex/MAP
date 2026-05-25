# MAP
MLDE-Enzyme-Activity-Prediction
This Repository contains the produced results of the authors Master Thesis.
Create an Environment using the MAP.yml and additionally install a suitable Pytorch configuration for your environment.
For this thesis, the script MAP/Code/12_MAP_Benchmark_ZS_setup.py has been applied as the main script.
It was initiated via Slurm scripts. One example herefore is provided: MAP/Code/120_run_benchmark_urz_single.sh
It simulates an MLDE experiment with a ZS-guided starting point selection.
MAP/Code/11_MAP_Benchmark_man_setup.py represents the non_ZS guided version for starting point selection, which was discussed in the discussions section.
Both scripts call the source components for on the fly-embedding generation, a model-wrapper class and another class for a sequential hyperparameter optimizer.
The folder also provides the additional script to generate ESMC embeddings.
