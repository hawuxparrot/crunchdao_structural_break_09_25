import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
import warnings
import gc
import pandas as pd
import os
import typing
import joblib
from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_all_gpu_memory_and_objects():
    """
    Attempts to aggressively clear GPU memory and delete PyTorch-related objects
    from the global namespace in a Jupyter/IPython environment.
    """

    known_global_vars_to_delete = [
        'model', 'optimizer', 'criterion', 'train_loader', 'val_loader',
        'full_dataset', 'train_dataset', 'val_dataset', 'test_loader',
        'my_model_instance',
    ]
    
    for var_name in known_global_vars_to_delete:
        if var_name in globals():
            try:
                globals()[var_name] = None
                del globals()[var_name]
                print(f"  - Deleted global variable: {var_name}")
            except NameError:
                pass
            except Exception as e:
                print(f"  - Error deleting global {var_name}: {e}")

    deleted_count = 0
    for var_name, obj in list(globals().items()):
        if isinstance(obj, (torch.Tensor, torch.nn.Module, torch.utils.data.Dataset, torch.utils.data.DataLoader)):
            try:
                globals()[var_name] = None
                del globals()[var_name]
                deleted_count += 1
            except Exception as e:
                pass
    if deleted_count > 0:
        print(f"  - Aggressively deleted {deleted_count} PyTorch objects from global scope.")

    gc.collect()
    print("  - Python garbage collector called.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  - PyTorch CUDA cache cleared.")
    else:
        print("  - Not using CUDA, no cache to clear.")


class TimeSeriesDataset(Dataset):
    def __init__(self, X_data: pd.DataFrame, y_data:typing.Optional[pd.Series], max_len_0: int, max_len_1: int, is_training: bool = False):
        self.max_len_0 = max_len_0
        self.max_len_1 = max_len_1
        self.ids = X_data.index.get_level_values('id').unique().tolist()
        self.X_data_grouped = X_data.groupby('id')
        self.y_data = y_data.loc[self.ids].values.astype(np.float32) if y_data is not None else None
        self.is_training = is_training # Added flag for augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        series_id = self.ids[idx]
        single_series_data = self.X_data_grouped.get_group(series_id)
        period_0_values = single_series_data[single_series_data['period'] == 0]['value'].values
        period_1_values = single_series_data[single_series_data['period'] == 1]['value'].values

        stats_0 = self._get_statistical_features(period_0_values)
        stats_1 = self._get_statistical_features(period_1_values)
        statistical_features = torch.tensor(np.concatenate([stats_0, stats_1]), dtype=torch.float32)
        # z-score normalization
        if len(period_0_values) > 1 and np.std(period_0_values) != 0:
            period_0_normalized = (period_0_values - np.mean(period_0_values)) / np.std(period_0_values)
        else:
            period_0_normalized = np.zeros_like(period_0_values, dtype=np.float32)

        if len(period_1_values) > 1 and np.std(period_1_values) != 0:
            period_1_normalized = (period_1_values - np.mean(period_1_values)) / np.std(period_1_values)
        else:
            period_1_normalized = np.zeros_like(period_1_values, dtype=np.float32)

        """
        if self.is_training:
            jitter_strength = 0.05 # Adjust as needed (e.g., 5% of a typical std)
            period_0_normalized += np.random.normal(0, jitter_strength, period_0_normalized.shape).astype(np.float32)
            period_1_normalized += np.random.normal(0, jitter_strength, period_1_normalized.shape).astype(np.float32)
        """

        # truncate if segment is longer than max_len
        truncated_0 = period_0_normalized[:self.max_len_0]
        truncated_1 = period_1_normalized[:self.max_len_1]
        pad_width_0 = self.max_len_0 - len(truncated_0)
        pad_width_1 = self.max_len_1 - len(truncated_1)
        
        padded_0 = np.pad(truncated_0, (0, pad_width_0), 'constant', constant_values=0).astype(np.float32)
        padded_1 = np.pad(truncated_1, (0, pad_width_1), 'constant', constant_values=0).astype(np.float32)

        tensor_0 = torch.tensor(padded_0, dtype=torch.float32).unsqueeze(1) # (max_len_0, 1)
        tensor_1 = torch.tensor(padded_1, dtype=torch.float32).unsqueeze(1) # (max_len_1, 1)

        if self.y_data is not None:
            label = torch.tensor(self.y_data[idx], dtype=torch.float32)
            return tensor_0, tensor_1, statistical_features, label
        else:
            return tensor_0, tensor_1, statistical_features
    
    def _get_statistical_features(self, series_values):
        series_arr = np.array(series_values, dtype=np.float32)
        features = np.zeros(7, dtype=np.float32)
        if len(series_arr) == 0:
            return features
        features[0] = np.mean(series_arr)
        features[1] = np.std(series_arr) if len(series_arr) > 1 else 0.0 # MODIFIED LINE: Explicit check for std
        features[2] = np.median(series_arr)
        features[3] = np.min(series_arr)
        features[4] = np.max(series_arr)
        if len(series_arr) > 2: # Skew requires at least 3 points
            features[5] = stats.skew(series_arr)
        if len(series_arr) > 3:
            features[6] = stats.kurtosis(series_arr)
        # might have to handle nans and infs
        return features


class CNN_RNN_Model(nn.Module):
    def __init__(self, max_len_0: int, max_len_1: int):
        super(CNN_RNN_Model, self).__init__()

        def cnn_block():
            return nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(), # Changed from GELU to ReLU
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(), # Changed from GELU to ReLU
                nn.MaxPool1d(kernel_size=2),

                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256),
                # No activation after the last Conv1D in the image (or it's the output layer for this block)
                nn.ReLU(), # Added ReLU here to be consistent with common CNN block design, though image shows output directly
                nn.MaxPool1d(kernel_size=2)
            )
        
        self.cnn_branch_0 = cnn_block()
        self.cnn_branch_1 = cnn_block()

        # Dummy input/output calculations need to be updated after changing CNN block
        dummy_input_0 = torch.zeros(1, 1, max_len_0) 
        dummy_output_0 = self.cnn_branch_0(dummy_input_0)
        
        dummy_input_1 = torch.zeros(1, 1, max_len_1)
        dummy_output_1 = self.cnn_branch_1(dummy_input_1) 
        
        # changed GRU to LSTM
        self.rnn_branch_0 = nn.LSTM(input_size=dummy_output_0.shape[1], hidden_size=128, bidirectional=True, batch_first=True)
        self.rnn_branch_1 = nn.LSTM(input_size=dummy_output_1.shape[1], hidden_size=128, bidirectional=True, batch_first=True)

        # simplified Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 128), # Reduced intermediate layer size
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.6), # Maintain high dropout

            nn.Linear(128, 64), # Another reduction
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.6), # Maintain high dropout

            nn.Linear(64, 1), # Final output layer
        )

    def forward(self, x0, x1):
        x0 = x0.permute(0, 2, 1) 
        x1 = x1.permute(0, 2, 1) 

        x0 = self.cnn_branch_0(x0) 
        x1 = self.cnn_branch_1(x1)

        x0 = x0.permute(0, 2, 1)
        x1 = x1.permute(0, 2, 1)

        # adjusted for LSTM output: LSTM returns (output, (h_n, c_n))
        _, (h0, _) = self.rnn_branch_0(x0)
        _, (h1, _) = self.rnn_branch_1(x1)

        h0_combined = torch.cat((h0[-2, :, :], h0[-1, :, :]), dim=1) 
        h1_combined = torch.cat((h1[-2, :, :], h1[-1, :, :]), dim=1)

        merged = torch.cat((h0_combined, h1_combined), dim=1) 

        output_logits = self.classifier(merged) 
        return output_logits.squeeze(1)

class HybridModel(nn.Module):
    def __init__(self, max_len_0: int, max_len_1: int, num_statistical_features: int):
        super(HybridModel, self).__init__()

        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding='same'), # Fewer filters
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        dummy_cnn_output_0 = self.cnn_layer(torch.zeros(1, 1, max_len_0))
        
        self.lstm_layer = nn.LSTM(input_size=dummy_cnn_output_0.shape[1], hidden_size=16, bidirectional=True, batch_first=True) # Smaller hidden size
        classifier_input_size = (16 * 2 * 2) + num_statistical_features

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 64), # Adjusted input size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5), # High dropout still relevant

            nn.Linear(64, 32), # Optional intermediate layer
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 1), # Final output layer
        )

    def forward(self, x0, x1, statistical_features):
        # process raw time series through CNN-LSTM branches
        x0_cnn = self.cnn_layer(x0.permute(0, 2, 1))
        x1_cnn = self.cnn_layer(x1.permute(0, 2, 1))
        
        x0_cnn = x0_cnn.permute(0, 2, 1)
        x1_cnn = x1_cnn.permute(0, 2, 1)

        # LSTM output: returns (output, (h_n, c_n))
        _, (h0, _) = self.lstm_layer(x0_cnn) 
        _, (h1, _) = self.lstm_layer(x1_cnn)

        # concatenate final hidden states from both directions for both branches
        h0_combined = torch.cat((h0[-2, :, :], h0[-1, :, :]), dim=1) # Last two layers for bidirectional
        h1_combined = torch.cat((h1[-2, :, :], h1[-1, :, :]), dim=1)

        # concatenate learned features from CNN-LSTM with explicit statistical features
        merged_nn_features = torch.cat((h0_combined, h1_combined), dim=1)
        final_features = torch.cat((merged_nn_features, statistical_features), dim=1)

        output_logits = self.classifier(final_features)
        return output_logits.squeeze(1)

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    MAX_POSSIBLE_LEN = 4096

    max_len_0 = 2048
    max_len_1 = 2048
    
    # pass is_training=True to the TimeSeriesDataset for augmentation
    full_dataset = TimeSeriesDataset(X_train, y_train, max_len_0, max_len_1, is_training=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count() // 2)

    val_dataset.dataset.is_training = False 
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count() // 2)

    num_statistical_features_total = 7 * 2
    model = HybridModel(max_len_0, max_len_1, num_statistical_features_total).to(DEVICE)
    print(model)

    # adjusted LR and increased weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3) # might need to adjust weight decay for regularization
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight.item():.2f}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', # Monitor validation ROC AUC (we want to maximize it)
        factor=0.5, # Reduce LR by 50%
        patience=5, # If Val ROC AUC doesn't improve for 5 epochs, reduce LR
        verbose=True,
        min_lr=1e-7 # Don't let LR go below this
    )

    best_val_roc_auc = -1.0
    patience_counter = 0
    PATIENCE = 20 # Increased early stopping patience to give more chance for slow improvements
    EPOCHS = 100 

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_idx, (x0_batch, x1_batch, stats_batch, labels_batch) in enumerate(train_loader):
            try:
                x0_batch, x1_batch, stats_batch, labels_batch = x0_batch.to(DEVICE), x1_batch.to(DEVICE), stats_batch.to(DEVICE), labels_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(x0_batch, x1_batch, stats_batch)

               
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"WARNING: NaN or Inf detected in model outputs at Epoch {epoch}, Batch {batch_idx}")
                    raise RuntimeError("NaN or Inf in model outputs")

                loss = criterion(outputs, labels_batch)

               
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN or Inf detected in loss at Epoch {epoch}, Batch {batch_idx}")
                    raise RuntimeError("NaN or Inf in loss calculation")

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Helps stabilize gradients

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"WARNING: NaN or Inf detected in gradients for '{name}' at Epoch {epoch}, Batch {batch_idx}")
                            raise RuntimeError(f"NaN or Inf in gradients for {name}")

                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_targets.extend(labels_batch.cpu().numpy())

            except Exception as e:
                print(f"\nERROR: Caught exception in Epoch {epoch}, Batch {batch_idx}: {type(e).__name__}: {e}")
                raise 

        avg_train_loss = train_loss / len(train_loader)
        train_roc_auc = roc_auc_score(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x0_batch, x1_batch, stats_batch, labels_batch in val_loader:
                x0_batch, x1_batch, stats_batch, labels_batch = x0_batch.to(DEVICE), x1_batch.to(DEVICE), stats_batch.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(x0_batch, x1_batch, stats_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_roc_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Train ROC AUC: {train_roc_auc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val ROC AUC: {val_roc_auc:.4f}")

        scheduler.step(val_roc_auc)

        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_directory_path, 'cnn_rnn_model_best.pth'))
            print(f"  --> New best model saved with Val ROC AUC: {best_val_roc_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break
        
        joblib.dump(
            {'max_len_0': max_len_0, 'max_len_1': max_len_1},
            os.path.join(model_directory_path, 'nn_params.joblib')
        )
        print("CNN/RNN model training complete and best model saved.")

def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    nn_params = joblib.load(os.path.join(model_directory_path, 'nn_params.joblib'))
    max_len_0 = nn_params['max_len_0']
    max_len_1 = nn_params['max_len_1']
    num_statistical_features = nn_params['num_statistical_features']
    model = HybridModel(max_len_0, max_len_1, num_statistical_features).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_directory_path, 'cnn_rnn_model_best.pth'), map_location=DEVICE))
    model.eval()

    yield  # Mark as ready

    for dataset in X_test:
        x0_tensor, x1_tensor, stats_tensor = TimeSeriesDataset(dataset, None, max_len_0, max_len_1)[0]
        x0_input = x0_tensor.unsqueeze(0).to(DEVICE)
        x1_input = x1_tensor.unsqueeze(0).to(DEVICE)
        stats_input = stats_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output_logits = model(x0_input, x1_input, stats_input)
            prediction = torch.sigmoid(output_logits).item()
        
        yield prediction