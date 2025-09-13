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
        'my_model_instance', # Add any other specific global names you might use
    ]
    
    for var_name in known_global_vars_to_delete:
        if var_name in globals():
            try:
                # Set to None first to break references, then delete
                globals()[var_name] = None
                del globals()[var_name]
                print(f"  - Deleted global variable: {var_name}")
            except NameError:
                pass # Already deleted or not found
            except Exception as e:
                print(f"  - Error deleting global {var_name}: {e}")

    deleted_count = 0
    for var_name, obj in list(globals().items()): # Use list() to avoid RuntimeError
        if isinstance(obj, (torch.Tensor, torch.nn.Module, torch.utils.data.Dataset, torch.utils.data.DataLoader)):\
            try:
                globals()[var_name] = None
                del globals()[var_name]
                deleted_count += 1
                # print(f"  - Aggressively deleted global PyTorch object: {var_name}") # Uncomment for verbose
            except Exception as e:
                # print(f"  - Error deleting aggressive {var_name}: {e}") # Uncomment for verbose
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
        
        self.y_data_labels = None
        if y_data is not None:
            self.y_data_labels = y_data.loc[self.ids].to_numpy().astype(np.float32)

        self.is_training = is_training

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        series_id = self.ids[idx]
        single_series_data = self.X_data_grouped.get_group(series_id)
        
        full_series_values = single_series_data['value'].values.astype(np.float32)
        if len(full_series_values) > 1 and np.std(full_series_values) != 0:
            full_series_normalized = (full_series_values - np.mean(full_series_values)) / np.std(full_series_values)
        else:
            full_series_normalized = np.zeros_like(full_series_values, dtype=np.float32)

        if self.is_training:
            jitter_strength = 0.05
            full_series_normalized += np.random.normal(0, jitter_strength, full_series_normalized.shape).astype(np.float32)
        
        period_0_mask = single_series_data['period'] == 0
        period_1_mask = single_series_data['period'] == 1

        period_0_normalized = full_series_normalized[period_0_mask]
        period_1_normalized = full_series_normalized[period_1_mask]

        raw_period_0_values = single_series_data[period_0_mask]['value'].values
        raw_period_1_values = single_series_data[period_1_mask]['value'].values
        
        stats_0 = self._get_statistical_features_extended(raw_period_0_values)
        stats_1 = self<strong>get_statistical_features_extended(raw_period_1_values)
        
        num_extended_stats = len(stats_0) # Dynamic size
        diff_stats = np.zeros(num_extended_stats, dtype=np.float32)
        for i in range(num_extended_stats):
            diff_stats[i] = stats_1[i] - stats_0[i]
        
        mean_0 = stats_0[0]
        std_0 = stats_0[1]
        var_0 = stats_0[7] # Assuming variance is now at index 7
        
        mean_1 = stats_1[0]
        std_1 = stats_1[1]
        var_1 = stats_1[7]

        mean_ratio = mean_1 / (mean_0 + 1e-6) if (mean_0 + 1e-6) != 0 else 0.0
        std_ratio = std_1 / (std_0 + 1e-6) if (std_0 + 1e-6) != 0 else 0.0
        var_ratio = var_1 / (var_0 + 1e-6) if (var_0 + 1e-6) != 0 else 0.0

        if len(raw_period_0_values) > 1 and len(raw_period_1_values) > 1:
            ks_stat, ks_p_val = stats.ks_2samp(raw_period_0_values, raw_period_1_values)
        else:
            ks_stat, ks_p_val = 0.0, 1.0
        
        if len(raw_period_0_values) > 0 and len(raw_period_1_values) > 0:
            try:
                mw_u_stat, mw_p_val = stats.mannwhitneyu(raw_period_0_values, raw_period_1_values, alternative='two-sided')
            except ValueError: # If all values are the same, mannwhitneyu might raise ValueError
                mw_u_stat, mw_p_val = 0.0, 1.0
        else:
            mw_u_stat, mw_p_val = 0.0, 1.0

        statistical_features = np.concatenate([
            stats_0, stats_1, diff_stats, 
            [mean_ratio, std_ratio, var_ratio, ks_stat, ks_p_val, mw_u_stat, mw_p_val]
        ])
        
        statistical_features = np.nan_to_num(statistical_features, nan=0.0, posinf=0.0, neginf=0.0)

        truncated_0 = period_0_normalized[:self.max_len_0]
        truncated_1 = period_1_normalized[:self.max_len_1]
        pad_width_0 = self.max_len_0 - len(truncated_0)
        pad_width_1 = self.max_len_1 - len(truncated_1)
        
        padded_0 = np.pad(truncated_0, (0, pad_width_0), 'constant', constant_values=0).astype(np.float32)
        padded_1 = np.pad(truncated_1, (0, pad_width_1), 'constant', constant_values=0).astype(np.float32)

        tensor_0 = torch.tensor(padded_0, dtype=torch.float32).unsqueeze(1)
        tensor_1 = torch.tensor(padded_1, dtype=torch.float32).unsqueeze(1)

        if self.y_data_labels is not None:
            label = torch.tensor(self.y_data_labels[idx], dtype=torch.float32)
            return tensor_0, tensor_1, torch.tensor(statistical_features, dtype=torch.float32), label
        else:
            return tensor_0, tensor_1, torch.tensor(statistical_features, dtype=torch.float32)
    
    def _get_statistical_features_extended(self, series_values):
        series_arr = np.array(series_values, dtype=np.float32)
        
        # initialize all features to 0.0 to handle empty or very short arrays gracefully
        features = np.zeros(9, dtype=np.float32) 
        
        if len(series_arr) == 0:
            return features
        
        features[0] = np.mean(series_arr)
        features[1] = np.std(series_arr) if len(series_arr) > 1 else 0.0
        features[2] = np.median(series_arr)
        features[3] = np.min(series_arr)
        features[4] = np.max(series_arr)
        
        if len(series_arr) > 2: # skewness requires at least 3 data points
            features[5] = stats.skew(series_arr)
        else:
            features[5] = 0.0 # default skew to 0 for very short series
            
        if len(series_arr) > 3: # kurtosis requires at least 4 data points
            features[6] = stats.kurtosis(series_arr)
        else:
            features[6] = 0.0 # default kurtosis to 0 for very short series

        features[7] = np.var(series_arr) if len(series_arr) > 1 else 0.0
        features[8] = stats.iqr(series_arr) if len(series_arr) > 1 else 0.0
        
        # replace NaNs or Infs that might arise from statistical calculations
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features


class HybridModel(nn.Module):
    def __init__(self, max_len_0: int, max_len_1: int, num_statistical_features: int):
        super(HybridModel, self).__init__()

        def cnn_block():
            return nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

        self.cnn_branch_0 = cnn_block()
        self.cnn_branch_1 = cnn_block()

        # dummy input/output calculations to get LSTM input size
        dummy_input_0 = torch.zeros(1, 1, max_len_0) 
        dummy_output_0 = self.cnn_branch_0(dummy_input_0)
        
        dummy_input_1 = torch.zeros(1, 1, max_len_1)
        dummy_output_1 = self.cnn_branch_1(dummy_input_1)
        
        # num_layers for LSTMs is 3
        # hidden_size is 256
        self.lstm_branch_0 = nn.LSTM(input_size=dummy_output_0.shape[1], hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)
        self.lstm_branch_1 = nn.LSTM(input_size=dummy_output_1.shape[1], hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)

        # classifier input size: (hidden_size * 2 for bidir * 2 for two branches) 
        # + (hidden_size * 2 for the explicit difference feature) + num_statistical_features
        classifier_input_size = (256 * 2 * 2) + (256 * 2) + num_statistical_features

        # Changed: Increased Linear layer sizes, increased dropout to 0.5
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 1024), 
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 1),
        )

    def forward(self, x0, x1, statistical_features):
        x0_cnn = self.cnn_branch_0(x0.permute(0, 2, 1))
        x1_cnn = self.cnn_branch_1(x1.permute(0, 2, 1))
        
        x0_cnn = x0_cnn.permute(0, 2, 1)
        x1_cnn = x1_cnn.permute(0, 2, 1)

        _, (h0, _) = self.lstm_branch_0(x0_cnn) 
        _, (h1, _) = self.lstm_branch_1(x1_cnn)

        h0_combined = torch.cat((h0[-2, :, :], h0[-1, :, :]), dim=1) 
        h1_combined = torch.cat((h1[-2, :, :], h1[-1, :, :]), dim=1)

        abs_diff_h_combined = torch.abs(h0_combined - h1_combined)

        merged_nn_features = torch.cat((h0_combined, h1_combined, abs_diff_h_combined), dim=1)
        final_features = torch.cat((merged_nn_features, statistical_features), dim=1)

        output_logits = self.classifier(final_features)
        return output_logits.squeeze(1)

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    max_len_0 = 1024
    max_len_1 = 1024
    
    temp_dataset = TimeSeriesDataset(X_train.head(200), y_train.head(200), max_len_0, max_len_1, is_training=False)
    _, _, temp_stats_features, _ = temp_dataset[0]
    num_statistical_features_total = temp_stats_features.shape[0]
    del temp_dataset, temp_stats_features

    full_dataset = TimeSeriesDataset(X_train, y_train, max_len_0, max_len_1, is_training=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count() // 2)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count() // 2)

    model = HybridModel(max_len_0, max_len_1, num_statistical_features_total).to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight.item():.2f}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=20, 
        verbose=True,
        min_lr=1e-7 
    )

    best_val_roc_auc = -1.0
    patience_counter = 0
    PATIENCE = 50 
    EPOCHS = 100 

    gradient_accumulation_steps = 8 

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        optimizer.zero_grad() 
        for batch_idx, (x0_batch, x1_batch, stats_batch, labels_batch) in enumerate(train_loader):
            try:
                x0_batch, x1_batch, stats_batch, labels_batch = x0_batch.to(DEVICE), x1_batch.to(DEVICE), stats_batch.to(DEVICE), labels_batch.to(DEVICE)
                
                outputs = model(x0_batch, x1_batch, stats_batch)

               
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"WARNING: NaN or Inf detected in model outputs at Epoch {epoch}, Batch {batch_idx}")
                    raise RuntimeError("NaN or Inf in model outputs")

                loss = criterion(outputs, labels_batch)
                loss = loss / gradient_accumulation_steps 

               
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN or Inf detected in loss at Epoch {epoch}, Batch {batch_idx}")
                    raise RuntimeError("NaN or Inf in loss calculation")

                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * gradient_accumulation_steps
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_targets.extend(labels_batch.cpu().numpy())

            except Exception as e:
                print(f"\\nERROR: Caught exception in Epoch {epoch}, Batch {batch_idx}: {type(e).__name__}: {e}")
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
            torch.save(model.state_dict(), os.path.join(model_directory_path, 'hybrid_model_best.pth'))
            print(f"  --> New best model saved with Val ROC AUC: {best_val_roc_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break
        
        joblib.dump(
            {'max_len_0': max_len_0, 'max_len_1': max_len_1, 'num_statistical_features': num_statistical_features_total},
            os.path.join(model_directory_path, 'nn_params.joblib')
        )
        print("HybridModel training complete and best model saved.")

def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    nn_params = joblib.load(os.path.join(model_directory_path, 'nn_params.joblib'))
    max_len_0 = nn_params['max_len_0']
    max_len_1 = nn_params['max_len_1']
    num_statistical_features = nn_params['num_statistical_features']
    model = HybridModel(max_len_0, max_len_1, num_statistical_features).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_directory_path, 'hybrid_model_best.pth'), map_location=DEVICE))
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