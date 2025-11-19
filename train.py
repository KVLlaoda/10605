import os
import numpy as np
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from model import ConvNet
from model import train_one_epoch
from model import validate

BEST_BASE_MODEL_ACCURACY = 0.7972
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

def prune_linear_14(model, train_loader, val_loader, criterion, optimizer, device):
    threshold, cur_iter = 0, 10
    num_epochs = 200
    while True:
        best_val_accuracy = -1
        threshold = (cur_iter + 1) * 0.01
        print("CURRENT THRESHOLD:", threshold)

        log_path = f'weights/log_linear_14_{cur_iter}.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                best_val_accuracy = float(f.readline().strip().split()[-1])
        
        if cur_iter == 0:
            state_dict = torch.load(f'weights/model_weights_initial.pt', map_location='cpu')
        else:
            state_dict = torch.load(f'weights/model_weights_linear_14_{cur_iter - 1}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        unfrozen_layers = [CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_17]
        layer_to_freeze = LINEAR_14
        frozen_layers = []
        print_sparsity(model)

        ### prune parameters with weights below threshold (EXCLUDE BIAS)
        w_mask = layer_to_freeze.weight.abs() >= threshold if layer_to_freeze is not None else None
        w_mask = w_mask.to(device) if w_mask is not None else None
        freeze_layers(frozen_layers)
        unfreeze_layers(unfrozen_layers)

        with torch.no_grad():
            layer_to_freeze.weight *= w_mask
        
        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)

        print(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        print(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_accuracy /= 100.0

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%')
            
            if val_accuracy > best_val_accuracy:
                print(f"\nSaving model with val accuracy: {val_accuracy}\n")
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f'weights/model_weights_linear_14_{cur_iter}.pt', _use_new_zipfile_serialization=False)
                with open(f'weights/log_linear_14_{cur_iter}.txt', 'w') as f:
                    f.write(f"{best_val_accuracy}\n")
                
        state_dict = torch.load(f'weights/model_weights_linear_14_{cur_iter}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_accuracy /= 100.0
        sparsity = compute_sparsity(model)
        score = (val_accuracy + sparsity) / 2 if val_accuracy > 0.6 and sparsity > 0 else 0

        print(f"Final Sparsity: {sparsity*100:.2f}%, Val Acc: {val_accuracy*100:.4f}, Score: {score:.4f}")
        print("-------------------------------------------------\n")

        if best_val_accuracy < BEST_BASE_MODEL_ACCURACY - 0.01:
            print("Stopping retraining as accuracy dropped too low.")
            return

        cur_iter += 1

def prune_conv2d_9(model, train_loader, val_loader, criterion, optimizer, device):
    threshold, cur_iter = 0, 0
    num_epochs = 200
    while True:
        best_val_accuracy = -1
        threshold = (cur_iter + 1) * 0.01
        log_path = f'weights/log_conv2d_9_{cur_iter}.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                best_val_accuracy = float(f.readline().strip().split()[-1])
        
        if cur_iter == 0:
            directory = Path("weights")
            pattern = re.compile(r"model_weights_linear_14_(\d+)\.pt")  # capture the index
            max_idx = -1
            linear14_final_version = None

            for file_path in directory.iterdir():
                if file_path.is_file():
                    match = pattern.fullmatch(file_path.name)
                    if match:
                        idx = int(match.group(1))
                        if idx > max_idx:
                            max_idx = idx
                            linear14_final_version = file_path
            state_dict = torch.load(f'{linear14_final_version}', map_location='cpu')
        else:
            state_dict = torch.load(f'weights/model_weights_conv2d_9_{cur_iter - 1}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        unfrozen_layers = [CONV2D_1, CONV2D_3, CONV2D_7, LINEAR_17]
        layer_to_freeze = CONV2D_9
        frozen_layers = [LINEAR_14]
        print_sparsity(model)

        for name, param in model.named_parameters():
            layer_sparsity = torch.sum(param == 0).item() / param.numel()
            print(f"Layer: {name}\nSparsity: {layer_sparsity*100:.2f}%, Size: {param.numel()}")
        print("\n")

        ### prune parameters with weights below threshold (EXCLUDE BIAS)
        w_mask = layer_to_freeze.weight.abs() >= threshold if layer_to_freeze is not None else None
        w_mask = w_mask.to(device) if w_mask is not None else None
        unfreeze_layers(unfrozen_layers)
        freeze_layers(frozen_layers)

        with torch.no_grad():
            layer_to_freeze.weight *= w_mask
        
        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)

        print(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        print(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_accuracy /= 100.0

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%')
            if val_accuracy > best_val_accuracy:
                print(f"\nSaving model with val accuracy: {val_accuracy}\n")
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f'weights/model_weights_conv2d_9_{cur_iter}.pt', _use_new_zipfile_serialization=False)
                with open(f'weights/log_conv2d_9_{cur_iter}.txt', 'w') as f:
                    f.write(f"{best_val_accuracy}\n")
        state_dict = torch.load(f'weights/model_weights_conv2d_9_{cur_iter}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_accuracy /= 100.0
        sparsity = compute_sparsity(model)
        score = (val_accuracy + sparsity) / 2 if val_accuracy > 0.6 and sparsity > 0 else 0

        print(f"Final Sparsity: {sparsity*100:.2f}%, Val Acc: {val_accuracy*100:.4f}, Score: {score:.4f}")
        print("-------------------------------------------------\n")
        if best_val_accuracy < BEST_BASE_MODEL_ACCURACY - 0.01:
            print("Stopping retraining as accuracy dropped too low.")
            return

        cur_iter += 1


def prune_conv2d_7(model, train_loader, val_loader, criterion, optimizer, device):
    threshold, cur_iter = 0, 0
    num_epochs = 200
    while True:
        best_val_accuracy = -1
        threshold = (cur_iter + 1) * 0.01

        unfrozen_layers = [model.model[0], model.model[2], model.model[16]]
        layer_to_freeze = model.model[6]
        frozen_layers = [model.model[8], model.model[13]]
        print("CURRENT THRESHOLD:", threshold)

        log_path = f'weights/log_conv2d_7_{cur_iter}.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                best_val_accuracy = float(f.readline().strip().split()[-1])
        
        if cur_iter == 0:
            directory = Path("weights")
            pattern = re.compile(r"model_weights_conv2d_9_(\d+)\.pt")  # capture the index
            max_idx = -1
            conv2d_9_final_version = None

            for file_path in directory.iterdir():
                if file_path.is_file():
                    match = pattern.fullmatch(file_path.name)
                    if match:
                        idx = int(match.group(1))
                        if idx > max_idx:
                            max_idx = idx
                            conv2d_9_final_version = file_path
            state_dict = torch.load(f'{conv2d_9_final_version}', map_location='cpu')
        else:
            state_dict = torch.load(f'weights/model_weights_conv2d_7_{cur_iter - 1}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        unfrozen_layers = [CONV2D_1, CONV2D_3, LINEAR_17]
        layer_to_freeze = CONV2D_7
        frozen_layers = [CONV2D_9, LINEAR_14]
        print_sparsity(model)

        for name, param in model.named_parameters():
            layer_sparsity = torch.sum(param == 0).item() / param.numel()
            print(f"Layer: {name}\nSparsity: {layer_sparsity*100:.2f}%, Size: {param.numel()}")
        print("\n")

        ### prune parameters with weights below threshold (EXCLUDE BIAS)
        w_mask = layer_to_freeze.weight.abs() >= threshold if layer_to_freeze is not None else None
        w_mask = w_mask.to(device) if w_mask is not None else None
        freeze_layers(frozen_layers)
        unfreeze_layers(unfrozen_layers)

        with torch.no_grad():
            layer_to_freeze.weight *= w_mask
        
        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)

        print(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        print(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_accuracy /= 100.0

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            if val_accuracy > best_val_accuracy:
                print(f"\nSaving model with val accuracy: {val_accuracy}\n")
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f'weights/model_weights_conv2d_7_{cur_iter}.pt', _use_new_zipfile_serialization=False)
                with open(f'weights/log_conv2d_7_{cur_iter}.txt', 'w') as f:
                    f.write(f"{best_val_accuracy}\n")
        
        state_dict = torch.load(f'weights/model_weights_conv2d_7_{cur_iter}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_accuracy /= 100.0
        sparsity = compute_sparsity(model)
        score = (val_accuracy + sparsity) / 2 if val_accuracy > 0.6 and sparsity > 0 else 0

        print(f"Final Sparsity: {sparsity*100:.2f}%, Val Acc: {val_accuracy*100:.4f}, Score: {score:.4f}")
        print("-------------------------------------------------\n")
        if best_val_accuracy < BEST_BASE_MODEL_ACCURACY - 0.01:
            print("Stopping retraining as accuracy dropped too low.")
            return

        cur_iter += 1

def prune_conv2d_3(model, train_loader, val_loader, criterion, optimizer, device):
    best_base_model_accuracy = 0.7972
    threshold, cur_iter = 0, 0
    while True:
        best_val_accuracy = -1
        num_epochs, threshold = 200, (cur_iter + 1) * 0.01
        print("CURRENT THRESHOLD:", threshold)

        log_path = f'weights/log_conv2d_3_{cur_iter}.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                best_val_accuracy = float(f.readline().strip().split()[-1])
        
        if cur_iter == 0:
            directory = Path("weights")
            pattern = re.compile(r"model_weights_conv2d_7_(\d+)\.pt")  # capture the index
            max_idx = -1
            conv2d_7_final_version = None

            for file_path in directory.iterdir():
                if file_path.is_file():
                    match = pattern.fullmatch(file_path.name)
                    if match:
                        idx = int(match.group(1))
                        if idx > max_idx:
                            max_idx = idx
                            conv2d_7_final_version = file_path
            state_dict = torch.load(f'{conv2d_7_final_version}', map_location='cpu')
        else:
            state_dict = torch.load(f'weights/model_weights_conv2d_3_{cur_iter - 1}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        unfrozen_layers = [CONV2D_1, LINEAR_17]
        layer_to_freeze = CONV2D_3
        frozen_layers = [CONV2D_7, CONV2D_9, LINEAR_14]
        print_sparsity(model)

        for name, param in model.named_parameters():
            layer_sparsity = torch.sum(param == 0).item() / param.numel()
            print(f"Layer: {name}\nSparsity: {layer_sparsity*100:.2f}%, Size: {param.numel()}")
        print("\n")

        ### prune parameters with weights below threshold (EXCLUDE BIAS)
        w_mask = layer_to_freeze.weight.abs() >= threshold if layer_to_freeze is not None else None
        w_mask = w_mask.to(device) if w_mask is not None else None
        freeze_layers(frozen_layers)

        with torch.no_grad():
            layer_to_freeze.weight *= w_mask
        
        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)
        print(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        print(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_accuracy /= 100.0

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            if val_accuracy > best_val_accuracy:
                print(f"\nSaving model with val accuracy: {val_accuracy}\n")
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f'weights/model_weights_conv2d_3_{cur_iter}.pt', _use_new_zipfile_serialization=False)
                with open(f'weights/log_conv2d_7_{cur_iter}.txt', 'w') as f:
                    f.write(f"{best_val_accuracy}\n")
        
        state_dict = torch.load(f'weights/model_weights_conv2d_3_{cur_iter}.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_accuracy /= 100.0
        sparsity = compute_sparsity(model)
        score = (val_accuracy + sparsity) / 2 if val_accuracy > 0.6 and sparsity > 0 else 0

        print(f"Final Sparsity: {sparsity*100:.2f}%, Val Acc: {val_accuracy*100:.4f}, Score: {score:.4f}")
        print("-------------------------------------------------\n")
        if best_val_accuracy < BEST_BASE_MODEL_ACCURACY - 0.01:
            print("Stopping retraining as accuracy dropped too low.")
            return

        cur_iter += 1

def prune_linear_17(model, train_loader, val_loader, criterion, optimizer, device):
    pass

def prune_conv2d_1(model, train_loader, val_loader, criterion, optimizer, device):
    pass

def print_sparsity(model):
    for name, param in model.named_parameters():
        layer_sparsity = torch.sum(param == 0).item() / param.numel()
        print(f"Layer: {name}\nSparsity: {layer_sparsity*100:.2f}%, Size: {param.numel()}")
    print("\n")

def get_layers(model):
    return model.model[0], model.model[2], model.model[6], model.model[8], model.model[13], model.model[16]

def compute_sparsity(model):
    total_zeros = sum(torch.sum(p == 0).item() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    sparsity = total_zeros / total_params
    return sparsity

def freeze_layers(layers):
    for layer in layers:
        for name, param in layer.named_parameters():
            if "bias" in name:
                param.requires_grad = True   # allow bias training
            else:
                param.requires_grad = False  # freeze weights

def unfreeze_layers(layers):
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True

# load train and val
train_images = pickle.load(open('train_images.pkl', 'rb'))
train_labels = pickle.load(open('train_labels.pkl', 'rb'))
val_images = pickle.load(open('val_images.pkl', 'rb'))
val_labels = pickle.load(open('val_labels.pkl', 'rb'))
train_images = torch.tensor(train_images, dtype=torch.float32)
val_images = torch.tensor(val_images, dtype=torch.float32)
train_images = train_images.permute(0, 3, 1, 2)
val_images = val_images.permute(0, 3, 1, 2)
# During training or summary
train_images = train_images.to(device)
val_images = val_images.to(device)
train_dataset = TensorDataset(train_images,
                              torch.tensor(train_labels.squeeze(), dtype=torch.long))
val_dataset = TensorDataset(val_images,
                            torch.tensor(val_labels.squeeze(), dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

# prune_linear_14(model, train_loader, val_loader, criterion, optimizer, device)
prune_conv2d_9(model, train_loader, val_loader, criterion, optimizer, device)
prune_conv2d_7(model, train_loader, val_loader, criterion, optimizer, device)
# prune_conv2d_3(model, train_loader, val_loader, criterion, optimizer,
#               device)
# prune_linear_17(model, train_loader, val_loader, criterion, optimizer,
#               device)
# prune_conv2d_1(model, train_loader, val_loader, criterion, optimizer,
#               device)


