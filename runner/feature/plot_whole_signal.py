from runner.feature import base_plot_writer_runner
from utils.data import dataset

if __name__ == '__main__':
    WINDOW_SIZE = dataset.VbsDataSetFactory.SIGNAL_LENGTH
    STRIDE = WINDOW_SIZE

    base_plot_writer_runner.run(WINDOW_SIZE, STRIDE)
