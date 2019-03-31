from runner.feature import base_plot_writer_runner

if __name__ == '__main__':
    WINDOW_SIZE = 10000
    STRIDE = 5000

    base_plot_writer_runner.run(WINDOW_SIZE, STRIDE)
