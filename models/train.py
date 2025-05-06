import torch
import torch.nn as nn
import logging
from models import TimeSeriesTransformer, Encoder
from loss import cross_modal_contrastive_loss
from load_data import *
import torch.optim as optim


def train(train_loader, model, logger, epochs=50):
    # === 3. Train the model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.double()
    model.train()

    logger.info("Ready for training......")
    logger.info(f"Running on device {device}")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.double), batch_y.to(
                device
            )
            optimizer.zero_grad()
            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        accuracy = correct / total * 100
        logger.info(
            f"Epoch {epoch+1:02d}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.2f}%"
        )


def contrastive_train(
    sensor_encoder,
    gcms_encoder,
    dataloader,
    gcms_input_dim,
    sensor_input_dim,
    logger,
    embedding_dim=16,
    hidden_dim=128,
    temperature=0.07,
    num_epochs=100,
):
    # Put on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcms_encoder.to(device)
    sensor_encoder.to(device)

    # Define optimizer
    # We'll optimize both encoders' parameters together
    params = list(gcms_encoder.parameters()) + list(sensor_encoder.parameters())

    optimizer = optim.Adam(params, lr=1e-3)

    # Training loop
    for epoch in range(num_epochs):
        gcms_encoder.train()
        sensor_encoder.train()

        total_loss = 0.0
        for x_sensor, x_gcms in dataloader:
            x_gcms = x_gcms.to(device)
            x_sensor = x_sensor.to(device)

            optimizer.zero_grad()

            # Forward pass
            z_gcms = gcms_encoder(x_gcms)  # shape [batch_size, embedding_dim]
            z_sensor = sensor_encoder(x_sensor)

            # Contrastive loss
            loss = cross_modal_contrastive_loss(z_gcms, z_sensor, temperature)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return gcms_encoder, sensor_encoder


def representation_train(
    model,
    dataloader,
    gcms_input_dim,
    sensor_input_dim,
    logger,
    embedding_dim=16,
    hidden_dim=128,
    temperature=0.07,
    num_epochs=100,
):
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    # ==== Training Setup ====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    # ==== Training Loop ====
    for epoch in range(num_epochs):
        # Forward pass
        outputs, embeddings = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_outputs, _ = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")


if __name__ == "__main__":
    training_path = "/home/dewei/workspace/smell-net/training"
    testing_path = "/home/dewei/workspace/smell-net/testing"
    training_data, testing_data, min_len = load_data(training_path, testing_path)
    train_loader, le = prepare_tensors(training_data, min_len)
    model = TimeSeriesTransformer(
        input_dim=12, model_dim=64, num_classes=len(le.classes_)
    )
    train(train_loader, model)
