# SHA3 Experimentation Helper Scripts

## gen_sha3_sweep_configs.py
Place in the chipyard/scripts directory, and run to generate sweep configs to parameterize the SHA3 Rocket Config.
```bash
python gen_sha3_sweep_configs.py
```

## sweep_sha3_configs_benchmarks.sh
Place in the chipyard/scripts directory, and run to elaborate the Configs and run the benchmarks on each successfully built config

## extract_data.py
Move the successful config directories with the benchmark .log files to another directory, and put this script in the same directory. Run this script to convert the log file data to *.npy files for the model training.