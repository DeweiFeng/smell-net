from models import TimeSeriesTransformer, Encoder
from load_data import load_sensor_data, prepare_transformer_tensors
from train import train
from evaluate import transformer_evaluate
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
    real_time_testing_path = "/home/dewei/workspace/smell-net/real_time_testing"

    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(training_path, testing_path, real_time_testing_path=real_time_testing_path)

    data_loader, le = prepare_transformer_tensors(training_data, min_len)

    model = TimeSeriesTransformer(
        input_dim=12, model_dim=64, num_classes=len(le.classes_)
    )

    train(data_loader, model, logger)

    transformer_evaluate(model, testing_data, le, logger)

    transformer_evaluate(model, real_time_testing_data, le, logger)


if __name__ == "__main__":
    main()
