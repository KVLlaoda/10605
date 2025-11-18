import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import ConvNet
from model import train_one_epoch
from model import validate

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

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

for r1 in np.arange(0.0, 1.0, 0.1):
    for r2 in np.arange(0.0, 1.0, 0.1):
        for r3 in np.arange(0.0, 1.0, 0.1):
            print(f"Dropout rates: {r1}, {r2}, {r3}")
            model = ConvNet(r1, r2, r3)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

            ### LOAD THE CORRECT MODEL HERE!
            continue_from_prev_training_run = True
            if continue_from_prev_training_run:
                state_dict = torch.load('my_model_weights_2.pt', map_location='cpu')
            else:
                state_dict = torch.load('best_sparsity_model.pt', map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device)

            for name, param in model.named_parameters():
                layer_sparsity = torch.sum(param == 0).item() / param.numel()
                print(f"Layer: {name}\nSparsity: {layer_sparsity*100:.2f}%, Size: {param.numel()}")
            print("\n")

            # ### prune low magnitude weights
            threshold = 0.01
            unfrozen_layers = [model.model[0], model.model[2], model.model[6], model.model[8], model.model[13], model.model[16]]
            layer_to_freeze = None
            # frozen_layers = [model.model[6], model.model[2], model.model[8], model.model[13]]
            frozen_layers = []

            w_mask = layer_to_freeze.weight.abs() >= threshold if layer_to_freeze is not None else None
            b_mask = layer_to_freeze.bias.abs() >= threshold if layer_to_freeze is not None else None
            w_mask = w_mask.to(device) if w_mask is not None else None
            b_mask = b_mask.to(device) if b_mask is not None else None
            for layer in unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for layer in frozen_layers:
                for name, param in layer.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True   # allow bias training
                    else:
                        param.requires_grad = False  # freeze weights

            if layer_to_freeze is not None:
                with torch.no_grad():
                    layer_to_freeze.weight *= w_mask
                frozen_layer_zeros = 0
                frozen_layer_params = 0
                for param in layer_to_freeze.parameters():
                    frozen_layer_zeros += torch.sum(param == 0).item()
                    frozen_layer_params += param.numel()
                print(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params}\n")

            total_zeros = 0
            total_params = 0
            for param in model.parameters():
                total_zeros += torch.sum(param == 0).item()
                total_params += param.numel()
            sparsity = total_zeros / total_params
            print(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

            ### now, retrain
            num_epochs = 50
            best_val_accuracy = -1
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")

                # Training
                train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, b_mask, layer_to_freeze)

                # Validation
                val_loss, val_accuracy = validate(model, val_loader, criterion, device)

                # Print epoch results
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
                if val_accuracy > best_val_accuracy:
                  print(f"\nSaving model with val accuracy: {val_accuracy}\n")
                  best_val_accuracy = val_accuracy
                  torch.save(model.state_dict(), 'best_training_model.pt', _use_new_zipfile_serialization=False)

            ### PRINT STATS
            state_dict = torch.load('best_training_model.pt', map_location='cpu')
            total_zeros = 0
            total_params = 0
            for param in model.parameters():
                total_zeros += torch.sum(param == 0).item()
                total_params += param.numel()
            sparsity = total_zeros / total_params
            model.load_state_dict(state_dict)
            model = model.to(device)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_accuracy /= 100.0
            if val_accuracy > 0.6 and sparsity > 0:
                score = (val_accuracy + sparsity) / 2
            else:
                score = 0
            print(f"Sparsity: {sparsity*100:.2f}%, Val Acc: {val_accuracy:.4f}, Score: {score:.4f}")
            print("-------------------------------------------------\n")