import os
import typing
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences_before, sequences_after, labels):
        self.sequences_before = [torch.tensor(s, dtype=torch.float32) for s in sequences_before]
        self.sequences_after = [torch.tensor(s, dtype=torch.float32) for s in sequences_after]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences_before[idx], self.sequences_after[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        seqs_before, seqs_after, labels = zip(*batch)
        padded_before = pad_sequence(seqs_before, batch_first=True, padding_value=0.0).unsqueeze(-1)
        padded_after = pad_sequence(seqs_after, batch_first=True, padding_value=0.0).unsqueeze(-1)
        return padded_before, padded_after, torch.stack(labels)

class SiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        _, (hidden, _) = self.encoder(x)
        if self.encoder.bidirectional:
            return torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            return hidden[-1]

    def forward(self, x1, x2):
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        combined = torch.cat((embedding1, embedding2), dim=1)
        return self.classifier(combined)


def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    model = SiameseLSTM(
        input_size=1, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    NUM_EPOCHS = 10
   
    for epoch in range(NUM_EPOCHS):
        model.train()
        for before_batch, after_batch, labels_batch in train_loader:
            before_batch, after_batch, labels_batch = before_batch.to(device), after_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(before_batch, after_batch)
            loss = criterion(outputs.squeeze(), labels_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for before_batch, after_batch, labels_batch in val_loader:
            before_batch, after_batch, labels_batch = before_batch.to(device), after_batch.to(device), labels_batch.to(device)
            outputs = model(before_batch, after_batch)
            loss = criterion(outputs.squeeze(), labels_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seqs_before, seqs_after = [], []
    grouped = X_train.groupby('id')
    for series_id in y_train.index:
        group = grouped.get_group(series_id)
        seqs_before.append(group[group['period'] == 0]['value'].values.reshape(-1, 1))
        seqs_after.append(group[group['period'] == 1]['value'].values.reshape(-1, 1))
    labels = y_train.values.astype(float)

    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    seqs_before_train, seqs_after_train, labels_train = [seqs_before[i] for i in train_indices], [seqs_after[i] for i in train_indices], labels[train_indices]
    seqs_before_val, seqs_after_val, labels_val = [seqs_before[i] for i in val_indices], [seqs_after[i] for i in val_indices], labels[val_indices]

    scaler = StandardScaler().fit(np.vstack([s for s in seqs_before_train if s.shape[0] > 0]))
    
    train_dataset = PairedTimeSeriesDataset([scaler.transform(s) for s in seqs_before_train], [scaler.transform(s) for s in seqs_after_train], labels_train)
    val_dataset = PairedTimeSeriesDataset([scaler.transform(s) for s in seqs_before_val], [scaler.transform(s) for s in seqs_after_val], labels_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=PairedTimeSeriesDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=PairedTimeSeriesDataset.collate_fn)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=30) 
    best_params = study.best_trial.params
    print("Best trial found:", best_params)

    print("\nTraining final model...")
    seqs_before_scaled = [scaler.transform(s) for s in seqs_before]
    seqs_after_scaled = [scaler.transform(s) for s in seqs_after]
    full_dataset = PairedTimeSeriesDataset(seqs_before_scaled, seqs_after_scaled, labels)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, collate_fn=PairedTimeSeriesDataset.collate_fn)

    final_model = SiameseLSTM(1, **best_params).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
    criterion = nn.BCELoss()

    for epoch in range(15): 
        final_model.train()
        total_loss = 0
        for before, after, label_batch in full_loader:
            before, after, label_batch = before.to(device), after.to(device), label_batch.to(device)
            optimizer.zero_grad()
            outputs = final_model(before, after)
            loss = criterion(outputs.squeeze(), label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Final Training Epoch {epoch+1}/15, Loss: {total_loss/len(full_loader):.4f}")

    torch.save(final_model.state_dict(), os.path.join(model_directory_path, "model.pth"))
    joblib.dump(scaler, os.path.join(model_directory_path, "scaler.joblib"))
    joblib.dump(best_params, os.path.join(model_directory_path, "best_params.joblib"))
    print("Training complete and final model saved.")


def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
) -> typing.Generator[float, None, None]:
    
    best_params = joblib.load(os.path.join(model_directory_path, "best_params.joblib"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SiameseLSTM(input_size=1, **best_params)
    model.load_state_dict(torch.load(os.path.join(model_directory_path, "model.pth")))
    model.to(device)
    model.eval()
    
    scaler = joblib.load(os.path.join(model_directory_path, "scaler.joblib"))
    yield # Mark as ready

    for dataset in X_test:
        before_series = dataset[dataset['period'] == 0]['value'].values.reshape(-1, 1)
        after_series = dataset[dataset['period'] == 1]['value'].values.reshape(-1, 1)
        
        before_scaled = scaler.transform(before_series)
        after_scaled = scaler.transform(after_series)
        
        tensor_before = torch.tensor(before_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        tensor_after = torch.tensor(after_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(tensor_before, tensor_after)
            
        yield prediction.item()