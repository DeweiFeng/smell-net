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
from dateset import PairedDataset

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
    real_time_testing_path = "/home/dewei/workspace/smell-net/real_time_testing"
    gcms_path = "/home/dewei/workspace/smell-net/processed_full_gcms_dataframe.csv"

    Pair

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(training_path, testing_path, real_time_testing_path=real_time_testing_path, categories=["Nuts"])

    train_data_loader, le = process_data_regular(training_data)

    pair_data = []

    for i in range(train_data_loader):
        pair_data.append((smell_data[i], gcms_data[int(y[i])]))

    test_data_loader, _ = process_data_regular(testing_data, le)

    real_time_test_data_loader, _ = process_data_regular(real_time_testing_data, le)

    sensor_model = Encoder(input_dim=12)
    gcms_model = Encoder(input_dim=12)

    train(train_data_loader, model, logger, epochs=3)

    regular_evaluate(model, test_data_loader, le, logger)

    regular_evaluate(model, real_time_test_data_loader, le, logger)


if __name__ == "__main__":
    main()
