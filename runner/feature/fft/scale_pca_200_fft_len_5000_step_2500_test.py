from pathlib import Path

from runner.feature.standard_scaler_base_test_runner import run

if __name__ == '__main__':
    run(Path("/mnt/share/vbs-power-line-fault-detection/features/fft/"
             "/pca/fft_length_{}_stride_{}".format(5000, 2500)))
