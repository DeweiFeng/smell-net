from models import TimeSeriesTransformer, Encoder
from load_data import (
    load_sensor_data,
    prepare_transformer_tensors,
    process_data_regular,
)
from train import train
from evaluate import transformer_evaluate, regular_evaluate
import logging
import os
import time

log_dir = "/home/dewei/workspace/smell-net/logs"

log_file_path = os.path.join(log_dir, f"{time.time()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)


def main():
    # set up logging
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/smell-net/training"
    testing_path = "/home/dewei/workspace/smell-net/testing"
    real_time_testing_path = "/home/dewei/workspace/smell-net/processed_real_time_testing_spice"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(training_path, testing_path, real_time_testing_path=real_time_testing_path, categories=["Spices"])

    train_data_loader, le = process_data_regular(training_data, dropped_columns=["Benzene", "Altitude"])

    test_data_loader, _ = process_data_regular(testing_data, le, dropped_columns=["Benzene", "Altitude"])

    real_time_test_data_loader, _ = process_data_regular(real_time_testing_data, le, dropped_columns=["Benzene", "Altitude"])

    model = Encoder(input_dim=10)

    train(train_data_loader, model, logger, epochs=3)

    regular_evaluate(model, test_data_loader, le, logger)

    regular_evaluate(model, real_time_test_data_loader, le, logger)


if __name__ == "__main__":
    main()
