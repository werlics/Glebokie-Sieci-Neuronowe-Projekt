from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from torch import nn
import time
import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from network import CNNNetwork
import soundfile

import warnings

warnings.filterwarnings("ignore")


class UrbanSoundDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transformation, target_sample_rate, num_samples, device, folds):
        self.annotations = pd.read_csv(csv_file)

        self.annotations = self.annotations[self.annotations['fold'].isin(folds)]
        self.annotations = self.annotations.reset_index(drop=True)

        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_info = self.annotations.iloc[idx]
        file_name = audio_info["slice_file_name"]
        fold = f"fold{audio_info['fold']}"
        label = audio_info["classID"]
        file_path = os.path.join(self.audio_dir, fold, file_name)

        signal, sample_rate = torchaudio.load(file_path, normalize=True)
        signal = signal.to(self.device)

        # resampling
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate).to(self.device)
            signal = resampler(signal)

        # mix down do mono
        signal = torch.mean(signal, dim=0, keepdim=True)

        # przycinannie / padding
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        signal = self.transformation(signal)
        signal = torchaudio.transforms.AmplitudeToDB()(signal)
        return signal, label


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    loop = tqdm.tqdm(data_loader, desc="Train")

    for input, target in loop:
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, dim=1)
        correct_predictions += (predicted == target).sum().item()
        total_samples += target.size(0)

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(data_loader)
    acc = (correct_predictions / total_samples) * 100
    return avg_loss, acc


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = loss_fn(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)

    avg_loss = running_loss / len(data_loader)
    acc = (correct_predictions / total_samples) * 100
    return avg_loss, acc


def fit(model, train_loader, test_loader, loss_fn, optimiser, device, epochs, scheduler):
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    MIN_LR = 1e-5

    for i in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        scheduler.step(test_loss)
        current_lr = optimiser.param_groups[0]['lr']

        end_time = time.time()
        epoch_duration = end_time - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"\n   Epoch {i + 1}/{epochs} | Time: {epoch_duration:.1f}s | LR: {current_lr:.6f}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
        print("-" * 40)

        if current_lr < MIN_LR:
            print(f"Learning Rate ({current_lr}) spadł poniżej {MIN_LR}.")
            print("Model przestał sie uczyć")
            break

    return history


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # wykres z loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss", linestyle="--")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # wykres z acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc", linestyle="--")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "input/urbansound8k/UrbanSound8K.csv"
    audio_dir = "input/urbansound8k/"

    sample_rate = 22050
    num_samples = 22050*4

    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.001

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"sample_rate: {sample_rate}, num_samples: {num_samples}, BATCH_SIZE: {BATCH_SIZE}, EPOCHS: {EPOCHS}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=1024,hop_length=512,n_mels=128)

    train_dataset = UrbanSoundDataset(
        csv_file, audio_dir, mel_spectrogram, sample_rate, num_samples, device,
        folds=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    test_dataset = UrbanSoundDataset(
        csv_file, audio_dir, mel_spectrogram, sample_rate, num_samples, device,
        folds=[10]
    )

    print(f"Próbki treningowe: {len(train_dataset)}")
    print(f"Próbki testowe: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    sample_sig, _ = train_dataset[0]

    cnn = CNNNetwork(input_shape=sample_sig.shape).to(device)

    print(cnn)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode='min',
        factor=0.1,
        patience=3
    )

    # ------------------------------------------------  TRENING
    total_start = time.time()

    history = fit(cnn, train_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS, scheduler)

    total_end = time.time()
    print(f"Całkowity czas treningu: {(total_end - total_start) / 60:.2f} minut")

    torch.save(cnn.state_dict(), "model_urban8k.pth")
    print("Model zapisany")
    plot_history(history)