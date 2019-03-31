from runner.feature.fft.pca_base_runner import run

if __name__ == '__main__':
    run(n_components=200, fft_length=5000, fft_stride=2500)
