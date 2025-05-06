from load_data import load_gcms_data, load_sensor_data

if __name__ == "__main__":
    gcms_path = "/home/dewei/workspace/smell-net/full_gcms_dataframe.csv"
    training_path = "/home/dewei/workspace/smell-net/training"
    testing_path = "/home/dewei/workspace/smell-net/testing"

    training_data, testing_data, min_len = load_sensor_data(training_path, testing_path, contrastive_learning=True)
    # X_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)
    # create_pair_data