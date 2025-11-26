import os
import numpy as np
import pickle
import re
import shutil
import csv
import torch
import random
import logging
import pandas as pd
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

def prune_linear_14(model, train_loader, val_loader, criterion, optimizer, device, num_iters=50, num_epochs=5, experiment_idx=1, exp_revival=1.5):
    # Set up logging
    dir = f"experiments/exp_{experiment_idx}"
    open(f"{dir}/log.txt", 'w').close()
    log_file = f"{dir}/log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    cur_iter = -1
    while True:
        cur_iter += 1
        threshold = (cur_iter + 1) * 0.01
        logging.info(f"CURRENT THRESHOLD: {threshold}")
        logging.info(f"\n=== ITERATION {cur_iter}, THRESHOLD={threshold:.4f} ===")

        # Load previous weights
        if cur_iter == 0:
            state_dict = torch.load(f'{dir}/model.pt', map_location='cpu')
        else:
            prev_weight_path = f'{dir}/model_{cur_iter - 1}.pt'
            state_dict = torch.load(prev_weight_path, map_location='cpu')

        model.load_state_dict(state_dict)
        model = model.to(device)
        torch.save(model.state_dict(), f'{dir}/model_{cur_iter}.pt', _use_new_zipfile_serialization=False)

        # Identify layers
        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        frozen_layers = [CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_17]
        layer_to_freeze = LINEAR_14
        unfrozen_layers = []

        # Prune weights below threshold (exclude bias)
        frozen_layer_zeros_before = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        freeze_layers(frozen_layers)
        unfreeze_layers(unfrozen_layers)
        with torch.no_grad():
            weight = layer_to_freeze.weight
            zero_mask = (weight == 0)

            # probabilistic revival
            p = 1 - compute_layer_sparsity(layer_to_freeze)
            resurrect_mask = (torch.rand_like(weight) < p ** exp_revival) & zero_mask

            # Set resurrected weights to threshold
            weight[resurrect_mask] = threshold
            w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
            layer_to_freeze.weight *= w_mask
        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)
        logging.info(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        logging.info(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        num_removed = frozen_layer_zeros - frozen_layer_zeros_before
        if num_removed == 0:
            logging.info("No weights removed in this iteration. Increment iteration.")
            continue
        else:
            logging.info(f"Number of weights removed in this iteration: {num_removed}")


        # Load CSV if it exists, otherwise create new
        metrics_csv = f'{dir}/metrics.csv'
        columns = [
            'iteration', 'train_acc', 'val_acc', 'score', 'sparsity',
            'linear_14_sparsity', 'linear_17_sparsity', 'conv2d_1_sparsity', 'conv2d_3_sparsity', 'conv2d_7_sparsity', 'conv2d_9_sparsity', 
        ]

        # If file exists, try to read it
        if os.path.exists(metrics_csv):
            try:
                df = pd.read_csv(metrics_csv)
                if df.empty or set(df.columns) != set(columns):
                    df = pd.DataFrame({col: [] for col in columns})
                    df['iteration'] = df['iteration'].astype(int)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame({col: [] for col in columns})
                df['iteration'] = df['iteration'].astype(int)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame({col: [] for col in columns})
            df['iteration'] = df['iteration'].astype(int)

        max_score_iteration = -1
        max_score = -1
        best_val_acc = -1
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weight_path = f'{dir}/model_{cur_iter}.pt'
                torch.save(model.state_dict(), weight_path, _use_new_zipfile_serialization=False)

                mask = df['iteration'] == cur_iter
                score = (best_val_acc / 100.0 + sparsity) / 2 if best_val_acc / 100.0 > 0.6 and sparsity > 0 else 0
                row_data = {
                    'iteration': cur_iter,
                    'train_acc': round(train_acc, 4),
                    'val_acc': round(val_acc, 4),
                    'score': round(score, 4),
                    'sparsity': round(compute_sparsity(model), 4),
                    'linear_14_sparsity': round(compute_layer_sparsity(LINEAR_14), 4),
                    'linear_17_sparsity': round(compute_layer_sparsity(LINEAR_17), 4),
                    'conv2d_1_sparsity': round(compute_layer_sparsity(CONV2D_1), 4),
                    'conv2d_3_sparsity': round(compute_layer_sparsity(CONV2D_3), 4),
                    'conv2d_7_sparsity': round(compute_layer_sparsity(CONV2D_7), 4),
                    'conv2d_9_sparsity': round(compute_layer_sparsity(CONV2D_9), 4)
                }

                if max_score < score:
                    max_score = score
                    max_score_iteration = cur_iter
                if mask.any():
                    for col, val in row_data.items():
                        df.loc[mask, col] = val
                else:
                    df.loc[len(df)] = row_data  # simple append

                # Save CSV
                df.to_csv(metrics_csv, index=False)
        
        # Stop condition
        if cur_iter - 20 >= max_score_iteration:
            logging.info("No improvement in score for 20 iterations. Stopping retraining.")
            break

def prune_conv2d_9(model, train_loader, val_loader, criterion, optimizer, device, num_iters=50, num_epochs=5, experiment_idx=1, exp_revival=1.5):
    # Set up logging
    dir = f"experiments/exp_{experiment_idx}"
    open(f"{dir}/log.txt", 'w').close()
    log_file = f"{dir}/log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    cur_iter = -1
    while True:
        cur_iter += 1
        threshold = (cur_iter + 1) * 0.01
        logging.info(f"CURRENT THRESHOLD: {threshold}")
        logging.info(f"\n=== ITERATION {cur_iter}, THRESHOLD={threshold:.4f} ===")

        # Load previous weights
        if cur_iter == 0:
            state_dict = torch.load(f'{dir}/model.pt', map_location='cpu')
        else:
            prev_weight_path = f'{dir}/model_{cur_iter - 1}.pt'
            state_dict = torch.load(prev_weight_path, map_location='cpu')

        model.load_state_dict(state_dict)
        model = model.to(device)
        torch.save(model.state_dict(), f'{dir}/model_{cur_iter}.pt', _use_new_zipfile_serialization=False)

        # Identify layers
        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        frozen_layers = [CONV2D_1, LINEAR_14, CONV2D_7, CONV2D_9, LINEAR_17]
        layer_to_freeze = CONV2D_3
        unfrozen_layers = []

        # Prune weights below threshold (exclude bias)
        frozen_layer_zeros_before = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
        freeze_layers(frozen_layers)
        unfreeze_layers(unfrozen_layers)
        with torch.no_grad():
            weight = layer_to_freeze.weight
            zero_mask = (weight == 0)

            # probabilistic revival
            p = 1 - compute_layer_sparsity(layer_to_freeze)
            resurrect_mask = (torch.rand_like(weight) < p ** exp_revival) & zero_mask

            # set resurrected weights to threshold
            weight[resurrect_mask] = threshold
            w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
            layer_to_freeze.weight *= w_mask

        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)
        logging.info(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        logging.info(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        num_removed = frozen_layer_zeros - frozen_layer_zeros_before
        if num_removed == 0:
            logging.info("No weights removed in this iteration. Increment iteration.")
            continue
        else:
            logging.info(f"Number of weights removed in this iteration: {num_removed}")


        # Load CSV if it exists, otherwise create new
        metrics_csv = f'{dir}/metrics.csv'
        columns = [
            'iteration', 'train_acc', 'val_acc', 'score', 'sparsity',
            'linear_14_sparsity', 'linear_17_sparsity', 'conv2d_1_sparsity', 'conv2d_3_sparsity', 'conv2d_7_sparsity', 'conv2d_9_sparsity', 
        ]

        # If file exists, try to read it
        if os.path.exists(metrics_csv):
            try:
                df = pd.read_csv(metrics_csv)
                if df.empty or set(df.columns) != set(columns):
                    df = pd.DataFrame({col: [] for col in columns})
                    df['iteration'] = df['iteration'].astype(int)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame({col: [] for col in columns})
                df['iteration'] = df['iteration'].astype(int)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame({col: [] for col in columns})
            df['iteration'] = df['iteration'].astype(int)

        max_score_iteration = -1
        max_score = -1
        best_val_acc = -1
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weight_path = f'{dir}/model_{cur_iter}.pt'
                torch.save(model.state_dict(), weight_path, _use_new_zipfile_serialization=False)

                mask = df['iteration'] == cur_iter
                score = (best_val_acc / 100.0 + sparsity) / 2 if best_val_acc / 100.0 > 0.6 and sparsity > 0 else 0
                row_data = {
                    'iteration': cur_iter,
                    'train_acc': round(train_acc, 4),
                    'val_acc': round(val_acc, 4),
                    'score': round(score, 4),
                    'sparsity': round(compute_sparsity(model), 4),
                    'linear_14_sparsity': round(compute_layer_sparsity(LINEAR_14), 4),
                    'linear_17_sparsity': round(compute_layer_sparsity(LINEAR_17), 4),
                    'conv2d_1_sparsity': round(compute_layer_sparsity(CONV2D_1), 4),
                    'conv2d_3_sparsity': round(compute_layer_sparsity(CONV2D_3), 4),
                    'conv2d_7_sparsity': round(compute_layer_sparsity(CONV2D_7), 4),
                    'conv2d_9_sparsity': round(compute_layer_sparsity(CONV2D_9), 4)
                }

                if max_score < score:
                    max_score = score
                    max_score_iteration = cur_iter
                if mask.any():
                    for col, val in row_data.items():
                        df.loc[mask, col] = val
                else:
                    df.loc[len(df)] = row_data  # simple append

                # Save CSV
                df.to_csv(metrics_csv, index=False)
        
        # Stop condition
        if cur_iter - 20 >= max_score_iteration:
            logging.info("No improvement in score for 20 iterations. Stopping retraining.")
            break


def prune_conv2d_7(model, train_loader, val_loader, criterion, optimizer, device, num_iters=50, num_epochs=5, experiment_idx=1, exp_revival=1.5):
    dir = f"experiments/exp_{experiment_idx}"
    open(f"{dir}/log.txt", 'w').close()
    log_file = f"{dir}/log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    cur_iter = -1
    while True:
        cur_iter += 1
        threshold = (cur_iter + 1) * 0.01
        logging.info(f"CURRENT THRESHOLD: {threshold}")
        logging.info(f"\n=== ITERATION {cur_iter}, THRESHOLD={threshold:.4f} ===")

        # Load previous weights
        if cur_iter == 0:
            state_dict = torch.load(f'{dir}/model.pt', map_location='cpu')
        else:
            prev_weight_path = f'{dir}/model_{cur_iter - 1}.pt'
            state_dict = torch.load(prev_weight_path, map_location='cpu')

        model.load_state_dict(state_dict)
        model = model.to(device)
        torch.save(model.state_dict(), f'{dir}/model_{cur_iter}.pt', _use_new_zipfile_serialization=False)

        # Identify layers
        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        frozen_layers = [CONV2D_1, LINEAR_14, CONV2D_3, CONV2D_9, LINEAR_17]
        layer_to_freeze = CONV2D_7
        unfrozen_layers = []

        # Prune weights below threshold (exclude bias)
        frozen_layer_zeros_before = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
        freeze_layers(frozen_layers)
        unfreeze_layers(unfrozen_layers)
        with torch.no_grad():
            weight = layer_to_freeze.weight
            zero_mask = (weight == 0)

            # probabilistic revival
            p = 1 - compute_layer_sparsity(layer_to_freeze)
            resurrect_mask = (torch.rand_like(weight) < p ** exp_revival) & zero_mask

            # set resurrected weights to threshold
            weight[resurrect_mask] = threshold
            w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
            layer_to_freeze.weight *= w_mask

        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)
        logging.info(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        logging.info(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        num_removed = frozen_layer_zeros - frozen_layer_zeros_before
        if num_removed == 0:
            logging.info("No weights removed in this iteration. Increment iteration.")
            continue
        else:
            logging.info(f"Number of weights removed in this iteration: {num_removed}")


        # Load CSV if it exists, otherwise create new
        metrics_csv = f'{dir}/metrics.csv'
        columns = [
            'iteration', 'train_acc', 'val_acc', 'score', 'sparsity',
            'linear_14_sparsity', 'linear_17_sparsity', 'conv2d_1_sparsity', 'conv2d_3_sparsity', 'conv2d_7_sparsity', 'conv2d_9_sparsity', 
        ]

        # If file exists, try to read it
        if os.path.exists(metrics_csv):
            try:
                df = pd.read_csv(metrics_csv)
                if df.empty or set(df.columns) != set(columns):
                    df = pd.DataFrame({col: [] for col in columns})
                    df['iteration'] = df['iteration'].astype(int)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame({col: [] for col in columns})
                df['iteration'] = df['iteration'].astype(int)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame({col: [] for col in columns})
            df['iteration'] = df['iteration'].astype(int)

        max_score_iteration = -1
        max_score = -1
        best_val_acc = -1
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weight_path = f'{dir}/model_{cur_iter}.pt'
                torch.save(model.state_dict(), weight_path, _use_new_zipfile_serialization=False)

                mask = df['iteration'] == cur_iter
                score = (best_val_acc / 100.0 + sparsity) / 2 if best_val_acc / 100.0 > 0.6 and sparsity > 0 else 0
                row_data = {
                    'iteration': cur_iter,
                    'train_acc': round(train_acc, 4),
                    'val_acc': round(val_acc, 4),
                    'score': round(score, 4),
                    'sparsity': round(compute_sparsity(model), 4),
                    'linear_14_sparsity': round(compute_layer_sparsity(LINEAR_14), 4),
                    'linear_17_sparsity': round(compute_layer_sparsity(LINEAR_17), 4),
                    'conv2d_1_sparsity': round(compute_layer_sparsity(CONV2D_1), 4),
                    'conv2d_3_sparsity': round(compute_layer_sparsity(CONV2D_3), 4),
                    'conv2d_7_sparsity': round(compute_layer_sparsity(CONV2D_7), 4),
                    'conv2d_9_sparsity': round(compute_layer_sparsity(CONV2D_9), 4)
                }

                if max_score < score:
                    max_score = score
                    max_score_iteration = cur_iter
                if mask.any():
                    for col, val in row_data.items():
                        df.loc[mask, col] = val
                else:
                    df.loc[len(df)] = row_data  # simple append

                # Save CSV
                df.to_csv(metrics_csv, index=False)
        
        # Stop condition
        if cur_iter - 20 >= max_score_iteration:
            logging.info("No improvement in score for 20 iterations. Stopping retraining.")
            break

def prune_conv2d_3(model, train_loader, val_loader, criterion, optimizer, device, num_iters=50, num_epochs=5, experiment_idx=1, exp_revival=1.5):
    dir = f"experiments/exp_{experiment_idx}"
    open(f"{dir}/log.txt", 'w').close()
    log_file = f"{dir}/log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    cur_iter = -1
    while True:
        cur_iter += 1
        threshold = (cur_iter + 1) * 0.01
        logging.info(f"CURRENT THRESHOLD: {threshold}")
        logging.info(f"\n=== ITERATION {cur_iter}, THRESHOLD={threshold:.4f} ===")

        # Load previous weights
        if cur_iter == 0:
            state_dict = torch.load(f'{dir}/model.pt', map_location='cpu')
        else:
            prev_weight_path = f'{dir}/model_{cur_iter - 1}.pt'
            state_dict = torch.load(prev_weight_path, map_location='cpu')

        model.load_state_dict(state_dict)
        model = model.to(device)
        torch.save(model.state_dict(), f'{dir}/model_{cur_iter}.pt', _use_new_zipfile_serialization=False)

        # Identify layers
        CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
        frozen_layers = [CONV2D_1, LINEAR_14, CONV2D_7, CONV2D_9, LINEAR_17]
        layer_to_freeze = CONV2D_3
        unfrozen_layers = []

        # Prune weights below threshold (exclude bias)
        frozen_layer_zeros_before = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
        freeze_layers(frozen_layers)
        unfreeze_layers(unfrozen_layers)
        with torch.no_grad():
            weight = layer_to_freeze.weight
            zero_mask = (weight == 0)

            # probabilistic revival
            p = 1 - compute_layer_sparsity(layer_to_freeze)
            resurrect_mask = (torch.rand_like(weight) < p ** exp_revival) & zero_mask

            # set resurrected weights to threshold
            weight[resurrect_mask] = threshold
            w_mask = (layer_to_freeze.weight.abs() >= threshold).to(device)
            layer_to_freeze.weight *= w_mask

        frozen_layer_zeros = sum(torch.sum(p == 0).item() for p in layer_to_freeze.parameters())
        frozen_layer_params = sum(p.numel() for p in layer_to_freeze.parameters())
        sparsity = compute_sparsity(model)
        logging.info(f"FROZEN LAYER SPARSITY: {frozen_layer_zeros / frozen_layer_params:.4f}")
        logging.info(f"TOTAL SPARSITY: {sparsity*100:.2f}%\n")

        num_removed = frozen_layer_zeros - frozen_layer_zeros_before
        if num_removed == 0:
            logging.info("No weights removed in this iteration. Increment iteration.")
            continue
        else:
            logging.info(f"Number of weights removed in this iteration: {num_removed}")


        # Load CSV if it exists, otherwise create new
        metrics_csv = f'{dir}/metrics.csv'
        columns = [
            'iteration', 'train_acc', 'val_acc', 'score', 'sparsity',
            'linear_14_sparsity', 'linear_17_sparsity', 'conv2d_1_sparsity', 'conv2d_3_sparsity', 'conv2d_7_sparsity', 'conv2d_9_sparsity', 
        ]

        # If file exists, try to read it
        if os.path.exists(metrics_csv):
            try:
                df = pd.read_csv(metrics_csv)
                if df.empty or set(df.columns) != set(columns):
                    df = pd.DataFrame({col: [] for col in columns})
                    df['iteration'] = df['iteration'].astype(int)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame({col: [] for col in columns})
                df['iteration'] = df['iteration'].astype(int)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame({col: [] for col in columns})
            df['iteration'] = df['iteration'].astype(int)

        max_score_iteration = -1
        max_score = -1
        best_val_acc = -1
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weight_path = f'{dir}/model_{cur_iter}.pt'
                torch.save(model.state_dict(), weight_path, _use_new_zipfile_serialization=False)

                mask = df['iteration'] == cur_iter
                score = (best_val_acc / 100.0 + sparsity) / 2 if best_val_acc / 100.0 > 0.6 and sparsity > 0 else 0
                row_data = {
                    'iteration': cur_iter,
                    'train_acc': round(train_acc, 4),
                    'val_acc': round(val_acc, 4),
                    'score': round(score, 4),
                    'sparsity': round(compute_sparsity(model), 4),
                    'linear_14_sparsity': round(compute_layer_sparsity(LINEAR_14), 4),
                    'linear_17_sparsity': round(compute_layer_sparsity(LINEAR_17), 4),
                    'conv2d_1_sparsity': round(compute_layer_sparsity(CONV2D_1), 4),
                    'conv2d_3_sparsity': round(compute_layer_sparsity(CONV2D_3), 4),
                    'conv2d_7_sparsity': round(compute_layer_sparsity(CONV2D_7), 4),
                    'conv2d_9_sparsity': round(compute_layer_sparsity(CONV2D_9), 4)
                }

                if max_score < score:
                    max_score = score
                    max_score_iteration = cur_iter
                if mask.any():
                    for col, val in row_data.items():
                        df.loc[mask, col] = val
                else:
                    df.loc[len(df)] = row_data  # simple append

                # Save CSV
                df.to_csv(metrics_csv, index=False)
        
        # Stop condition
        if cur_iter - 20 >= max_score_iteration:
            logging.info("No improvement in score for 20 iterations. Stopping retraining.")
            break

def print_sparsity(model):
    for name, param in model.named_parameters():
        layer_sparsity = torch.sum(param == 0).item() / param.numel()
        print(f"Layer: {name}\nSparsity: {layer_sparsity*100:.2f}%, Size: {param.numel()}")
    print("\n")

def get_layers(model):
    return model.model[0], model.model[2], model.model[6], model.model[8], model.model[13], model.model[16]

def compute_layer_sparsity(layer):
    total_zeros = sum(torch.sum(p == 0).item() for p in layer.parameters())
    total_params = sum(p.numel() for p in layer.parameters())
    sparsity = total_zeros / total_params
    return sparsity

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

def restart_model(model):
    CONV2D_1, CONV2D_3, CONV2D_7, CONV2D_9, LINEAR_14, LINEAR_17 = get_layers(model)
    unfrozen_layers = [CONV2D_1, CONV2D_9, CONV2D_3, CONV2D_7, LINEAR_17, LINEAR_14]
    unfreeze_layers(unfrozen_layers)
    num_epochs = 300
    w_mask = None
    layer_to_freeze = None
    best_val_acc = -1
    for i in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, w_mask, layer_to_freeze)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_acc /= 100.0
        print(f'Epoch [{i+1}/{num_epochs}], '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            print(f"Saving best model with val acc: {best_val_acc*100:.2f}% at epoch {i+1}")
            torch.save(best_model, 'base_model.pt', _use_new_zipfile_serialization=False)
    return model

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
state_dict = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)
print_sparsity(model)

# okay so we have this good base model, how do we improve sparsity while maintaining accuracy?
num_experiments = 2
experiment_indices = list(range(num_experiments, num_experiments + 1))
random.shuffle(experiment_indices)  # random order
for i in experiment_indices:
    # Create experiment directory
    print(f"CURRENT EXPERIMENT: {i}")
    os.makedirs(f'experiments/exp_{i}', exist_ok=True)

    # Path to model
    model_path = f'experiments/exp_{i}/model.pt'
    if not os.path.exists(model_path):
        continue

    # Load model state
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Run pruning
    prune_linear_14(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=100,
        experiment_idx=i,
        exp_revival=1.5
    )

# prune_linear_14(model, train_loader, val_loader, criterion, optimizer, device)
# prune_conv2d_9(model, train_loader, val_loader, criterion, optimizer, device)
# prune_conv2d_7(model, train_loader, val_loader, criterion, optimizer, device)
# prune_conv2d_3(model, train_loader, val_loader, criterion, optimizer,
#               device)