from runner.feature.fft.fft_summary_base_runner import train_run

if __name__ == '__main__':
    train_run(window_size=5000, step_size=5000, fft_length=200, fft_stride=100)
