import sys
import os
import csv
import json
import random
import time

# 独立运行训练脚本时也只使用预留的物理卡 6；主循环传入的显式设置优先。
os.environ.setdefault(
    'CUDA_VISIBLE_DEVICES',
    os.environ.get(
        'NEBULA_TRAINING_GPUS',
        os.environ.get('NEBULA_TRAINING_GPU', '6'),
    ),
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from tqdm import tqdm
import argparse

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import C6TransNet
import config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_precision(device, requested):
    requested = requested.lower()
    if requested not in {'bf16', 'fp16', 'fp32'}:
        raise ValueError(f"Unsupported precision: {requested}")

    if device.type != 'cuda':
        return 'fp32', None

    if requested == 'bf16':
        if torch.cuda.is_bf16_supported():
            return 'bf16', torch.bfloat16
        print("Warning: BF16 is not supported by this GPU; falling back to FP16.")
        return 'fp16', torch.float16
    if requested == 'fp16':
        return 'fp16', torch.float16
    return 'fp32', None


def atomic_torch_save(payload, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, output_path)


def atomic_json_dump(payload, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, output_path)

class Connect6Dataset(Dataset):
    def __init__(self, csv_files):
        self.samples = []
        print("Loading data...")
        for f in csv_files:
            try:
                # Use standard csv module which is more robust to inconsistent column counts than pandas
                with open(f, 'r', encoding='utf-8', errors='replace') as csvfile:
                    reader = csv.reader(csvfile)
                    
                    for row in reader:
                        if not row: continue
                        
                        # Check if it looks like a header
                        # If the row contains "moves" and "winner", it's likely a header
                        if any('moves' in str(x).lower() for x in row) and any('winner' in str(x).lower() for x in row):
                            continue 
                        
                        try:
                            # Clean values
                            vals = [str(x).strip() for x in row if str(x).strip()]
                            
                            moves_str = None
                            winner_str = None
                            policies_str = None
                            bonuses_str = ""
                            
                            # Intelligent Content-Based Parsing
                            # 1. Find Winner (black/white/draw) - most distinct
                            winner_idx = -1
                            for i, v in enumerate(vals):
                                v_lower = v.lower()
                                if v_lower in ['black', 'white', 'draw']:
                                    winner_str = v_lower
                                    winner_idx = i
                                    break
                            
                            # If no winner found, skip this malformed row
                            if winner_str is None:
                                continue
                                
                            # 2. Find Moves
                            potential_moves = []
                            for i, v in enumerate(vals):
                                if i == winner_idx: continue
                                if ':' in v: continue # Likely policy
                                
                                # Distinguish moves from bonuses
                                is_bonus = False
                                try:
                                    first_elem = v.split(',')[0].strip()
                                    float(first_elem)
                                    if '.' in first_elem:
                                        is_bonus = True
                                except:
                                    pass
                                    
                                if not is_bonus:
                                    potential_moves.append(v)
                            
                            if potential_moves:
                                moves_str = max(potential_moves, key=len)
                            else:
                                moves_str = ""

                            # 3. Find Policy (contains ':')
                            for i, v in enumerate(vals):
                                if i == winner_idx: continue
                                if ':' in v:
                                    policies_str = v
                                    break
                            
                            # 4. Find Bonus (optional)
                            for i, v in enumerate(vals):
                                if i == winner_idx: continue
                                if v == moves_str: continue
                                if v == policies_str: continue
                                bonuses_str = v
                                break

                            # Process Winner Value
                            if winner_str == 'black':
                                winner_val = 1.0
                            elif winner_str == 'white':
                                winner_val = -1.0
                            else:
                                winner_val = 0.0
                                
                            if policies_str is None: policies_str = ""
                            if moves_str is None: moves_str = ""
                            
                            self.samples.append((moves_str, winner_val, policies_str, bonuses_str))
                            
                        except Exception as e:
                            continue
                        
            except Exception as e:
                print(f"Error reading file {f}: {e}")
                
        # === Prioritized Sampling (Long Game Bonus) ===
        # Increase weight for games with > 60 moves
        # User requested "just a little bit"
        long_games = []
        for s in self.samples:
            moves_str = s[0]
            # Estimate move count by counting semicolons or commas
            if isinstance(moves_str, str):
                # Format is M1;M2;... or M1,M2
                count = max(moves_str.count(','), moves_str.count(';')) + 1
                if count > 60:
                     # Add with probability 0.2 (1.2x weight roughly)
                    if np.random.rand() < 0.2:
                        long_games.append(s)
        
        if long_games:
            print(f"Prioritization: Added {len(long_games)} extra samples from long games (>60 moves).")
            self.samples.extend(long_games)
        
        # === Data Balancing ===
        black_wins = [s for s in self.samples if s[1] == 1.0]
        white_wins = [s for s in self.samples if s[1] == -1.0]
        draws = [s for s in self.samples if s[1] == 0.0]
        
        print(f"Original Distribution: Black: {len(black_wins)}, White: {len(white_wins)}, Draw: {len(draws)}")
        
        if len(black_wins) > 0 and len(white_wins) > 0:
            target_count = max(len(black_wins), len(white_wins))
            
            # Oversample White
            if len(white_wins) < target_count:
                import random
                # Calculate how many needed
                needed = target_count - len(white_wins)
                # Randomly sample with replacement
                extras = random.choices(white_wins, k=needed)
                white_wins.extend(extras)
                print(f"Oversampled White to {len(white_wins)}")
                
            # Oversample Black (unlikely if Black is strong, but for completeness)
            if len(black_wins) < target_count:
                import random
                needed = target_count - len(black_wins)
                extras = random.choices(black_wins, k=needed)
                black_wins.extend(extras)
                print(f"Oversampled Black to {len(black_wins)}")
                
            # Reconstruct samples
            self.samples = black_wins + white_wins + draws
            # Shuffle
            import random
            random.shuffle(self.samples)
            print(f"Balanced Dataset Size: {len(self.samples)}")
                
        print(f"Loaded {len(self.samples)} games (after balancing).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Prevent infinite recursion by using a loop
        # Try at most len(self) times to find a valid sample
        attempts = 0
        max_attempts = len(self)
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self)
            moves_str, final_winner, policies_str, bonuses_str = self.samples[current_idx]
            
            # Helper to parse move string
            def parse_move(m_str):
                m_str = m_str.strip().strip('"').strip("'") # Remove quotes
                if not m_str: return None
                # Try integer first
                try:
                    move = int(m_str)
                    return move if 0 <= move < 361 else None
                except ValueError:
                    # Try coordinate like 'j10'
                    try:
                        m_str = m_str.lower()
                        col_char = m_str[0]
                        row_str = m_str[1:]
                        if 'a' <= col_char <= 's' and row_str.isdigit():
                            c = ord(col_char) - ord('a')
                            r = int(row_str) - 1
                            if 0 <= r < 19 and 0 <= c < 19:
                                return r * 19 + c
                    except:
                        pass
                return None

            # Replay game to pick a random state
            moves = []
            if moves_str and str(moves_str).lower() != 'nan':
                # Handle potentially quoted string like '"m5,n6"'
                clean_moves_str = str(moves_str).strip().strip('"').strip("'")
                for x in clean_moves_str.split(','):
                    m = parse_move(x)
                    if m is not None:
                        moves.append(m)
            
            # If valid moves found, verify we have enough to train on
            if len(moves) > 0:
                # FOUND VALID SAMPLE
                break
            
            # If invalid, increment attempts and continue loop
            attempts += 1
            if attempts == 1:
                # Only print warning for the first failure to avoid spam
                print(f"Warning: Failed to parse moves from sample {current_idx}: {moves_str}")
        
        # If we failed all attempts (should be impossible unless dataset is empty/garbage)
        if attempts >= max_attempts:
             raise RuntimeError("Dataset contains ONLY invalid samples! Check data formatting.")

        policies_list = policies_str.split('|')
        
        # Parse bonuses
        bonuses = []
        if bonuses_str and bonuses_str.lower() != 'nan':
            try:
                bonuses = [float(x) for x in bonuses_str.split(',')]
            except:
                bonuses = [] # Fail safe
        
        # We want to train on ALL positions, but that makes dataset huge.
        # Standard RL: Sample ONE position per game per epoch? Or expand all?
        # If we expand all, memory might explode.
        # Let's sample a random position from the game to keep diversity high but memory low.
        # Or better: Pre-process into (Board, Policy, Value) tuples? 
        # Given 2000 games * 30 moves = 60,000 samples. Memory is fine.
        # But policies_str decoding is slow.
        
        # Let's pick a move index to train on.
        # IMPROVEMENT: Priority Sampling for Dense Rewards
        # If there are moves with non-zero bonuses (good or bad moves), we should prioritize learning them!
        # Otherwise, the random sampling might miss the few critical "tactical" moments.
        
        interesting_indices = []
        if len(bonuses) > 0:
            # Find indices where bonus is significant (e.g. != 0)
            # Bonuses array might be shorter than moves if game ended early or logic diff?
            limit = min(len(moves), len(bonuses))
            for i in range(limit):
                if abs(bonuses[i]) > 0.05: # Threshold for "interesting"
                    interesting_indices.append(i)
        
        # Sampling Strategy:
        # 80% chance to pick an interesting move (if any exist)
        # 20% chance to pick random move (to maintain distribution coverage)
        
        if interesting_indices and np.random.rand() < 0.8:
            move_idx = np.random.choice(interesting_indices)
        else:
            # Fallback to random
            if len(moves) == 0:
                return self.__getitem__((idx + 1) % len(self))
            move_idx = np.random.randint(0, len(moves))
        
        # Get bonus for this specific move
        current_bonus = 0.0
        if move_idx < len(bonuses):
            current_bonus = bonuses[move_idx]
        
        # Vectorized state reconstruction. The previous implementation replayed
        # every prefix twice in Python, which dominated DataLoader CPU time.
        board = np.zeros((19, 19), dtype=np.int8)
        if move_idx:
            prefix_moves = np.asarray(moves[:move_idx], dtype=np.int64)
            indices = np.arange(move_idx, dtype=np.int64)
            players = np.ones(move_idx, dtype=np.int8)
            white_mask = (indices > 0) & (((indices + 1) // 2) % 2 == 1)
            players[white_mask] = -1
            board.reshape(-1)[prefix_moves] = players
        
        # Who is to play at move_idx?
        if move_idx == 0: player_to_move = 1
        else:
            if ((move_idx+1)//2) % 2 == 1: player_to_move = -1
            else: player_to_move = 1

        # Target Policy (Extract BEFORE augmentation)
        policy_target = np.zeros(361, dtype=np.float32)
        if move_idx < len(policies_list):
            p_str = policies_list[move_idx]
            if p_str:
                for item in p_str.split(';'):
                    try:
                        k, v = item.split(':', 1)
                        k, v = int(k), float(v)
                        if 0 <= k < 361 and np.isfinite(v) and v >= 0:
                            policy_target[k] = v
                    except (TypeError, ValueError):
                        continue

        policy_sum = float(policy_target.sum())
        if policy_sum <= 0:
            policy_target[moves[move_idx]] = 1.0
        else:
            policy_target /= policy_sum
            
        # --- Robust Data Augmentation ---
        # Randomly apply Flip and Rotation (Dihedral Group D4)
        # This gives 8 possible symmetries, multiplying effective data by 8x.
        
        # 1. Random Flip (Left-Right)
        if np.random.rand() < 0.5:
            board = np.fliplr(board)
            # Policy flip
            p2d = policy_target.reshape(19, 19)
            p2d = np.fliplr(p2d)
            policy_target = p2d.flatten()
            
        # 2. Random Rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4) # Number of 90-degree rotations
        if k > 0:
            board = np.rot90(board, k).copy() # copy to solve negative stride issues
            # Policy rotation
            p2d = policy_target.reshape(19, 19)
            p2d = np.rot90(p2d, k).copy()
            policy_target = p2d.flatten()
            
        # Ensure memory layout is contiguous after numpy transforms
        board = np.ascontiguousarray(board)
        policy_target = np.ascontiguousarray(policy_target)
            
        # Construct Input Tensor (17, 19, 19)
        features = np.zeros((17, 19, 19), dtype=np.float32)
        if player_to_move == 1: # Black
            features[0] = (board == 1)
            features[1] = (board == -1)
            features[16] = 1.0
        else: # White
            features[0] = (board == -1)
            features[1] = (board == 1)
            features[16] = 0.0
            
        # Target Value (Relative to current player)
        # Base: final_winner * player_to_move # 1 if win, -1 if loss
        base_value = final_winner * player_to_move
        
        # Add Dense Reward Bonus
        # Strategy: 
        # If Bonus > 0 (Good move): Boost value towards 1.0
        # If Bonus < 0 (Bad move): Penalize value towards -1.0
        # We simply add them, and clamp to [-1, 1] (or let Tanh handle it, but MSE target should be bounded)
        
        # Example: 
        # Loss (-1) + Good Move (+0.3) = -0.7 (Less bad)
        # Win (1) + Bad Move (-0.2) = 0.8 (Less good)
        # Draw (0) + Good Move (+0.3) = 0.3 (Slight advantage)
        
        # User requested Sparse Reward (ignore bonus)
        value_target = base_value 
        # value_target = base_value + current_bonus
        
        # Clamp to valid range [-1, 1] 
        # (Though technically >1 pushes gradients stronger, let's clip for stability)
        value_target = np.clip(value_target, -1.0, 1.0)
        
        # --- Sample Weighting for Weak White ---
        sample_weight = 1.0
        # If White Won (final_winner == -1.0), give higher weight
        # Reduced from 2.0 to 1.5 to prevent loss explosion
        # User requested 1.35
        if final_winner == -1.0:
            sample_weight = 1.35
        
        # If White is playing, emphasize
        # Reduced from 1.2 to 1.1
        if player_to_move == -1:
            sample_weight *= 1.1
            
        return torch.from_numpy(features), torch.tensor(policy_target), torch.tensor(value_target, dtype=torch.float32), torch.tensor(sample_weight, dtype=torch.float32)

def train(args):
    seed = config.SEED + max(args.generation, 0)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision, amp_dtype = resolve_precision(device, args.precision)

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = bool(config.ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(config.ALLOW_TF32)
        torch.set_float32_matmul_precision('high' if config.ALLOW_TF32 else 'highest')
        print(f"Training on {torch.cuda.get_device_name(0)} with {precision.upper()} precision")
    else:
        print("CUDA is unavailable; training falls back to FP32 on CPU.")

    # 1. Data
    if args.data:
        if os.path.isdir(args.data):
            files = glob.glob(os.path.join(args.data, "*.csv"))
        else:
            files = [args.data]
    else:
        files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
        
    if not files:
        print("No training data found!")
        return
        
    dataset = Connect6Dataset(files)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    num_workers = max(0, int(config.DATALOADER_WORKERS))
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': config.BATCH_SIZE_TRAIN,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'worker_init_fn': seed_worker,
        'generator': loader_generator,
    }
    if num_workers > 0:
        loader_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': config.DATALOADER_PREFETCH_FACTOR,
        })
    dataloader = DataLoader(**loader_kwargs)

    # 2. Model
    model = C6TransNet(input_planes=17).to(device)
    checkpoint = None

    # Load checkpoint
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location='cpu')

        # Determine which state_dict to use
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 3. Optimizer
    optimizer_kwargs = {
        'lr': config.LEARNING_RATE,
        'weight_decay': 1e-4,
    }
    if device.type == 'cuda' and config.USE_FUSED_ADAMW:
        optimizer_kwargs['fused'] = True
    try:
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError) as e:
        print(f"Fused AdamW unavailable ({e}); using the standard implementation.")
        optimizer_kwargs.pop('fused', None)
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    if args.resume_optimizer and checkpoint and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Each generation starts a fresh short cosine schedule while retaining
            # Adam moments from the accepted incumbent.
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE
                param_group['initial_lr'] = config.LEARNING_RATE
            print("Restored optimizer moments from checkpoint.")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"Optimizer state is incompatible and will be reset: {e}")

    # Cosine Annealing Scheduler: Decays from LR to MIN_LR over args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=config.MIN_LEARNING_RATE,
    )
    use_grad_scaler = device.type == 'cuda' and precision == 'fp16'
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    if config.TORCH_COMPILE and hasattr(torch, 'compile') and not isinstance(model, nn.DataParallel):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='max-autotune')

    # TensorBoard Writer
    # 数据集分析和 CPU 单测无需安装 TensorBoard；只在真正训练时加载。
    from torch.utils.tensorboard import SummaryWriter

    run_name = args.run_name or f"gen_{args.generation}"
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, 'tensorboard', run_name))
    global_step = int(checkpoint.get('global_step', 0)) if isinstance(checkpoint, dict) else 0

    # 4. Training Loop
    model.train()

    all_epoch_metrics = []
    train_started_at = time.perf_counter()
    total_samples_seen = 0

    for epoch in range(args.epochs):
        epoch_started_at = time.perf_counter()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_acc1 = 0
        total_acc5 = 0
        total_entropy = 0
        total_value_mae = 0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for features, policy_target, value_target, sample_weights in pbar:
            features = features.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            value_target = value_target.to(device, non_blocking=True).unsqueeze(1)
            sample_weights = sample_weights.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            # Forward
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                policy_logits, _, value_pred = model(features)

                # Loss
                # Policy: Focal Loss
                # Force the model to focus on hard examples (where prediction is wrong)
                # FL(pt) = - (1 - pt)^gamma * log(pt)
                log_probs = torch.log_softmax(policy_logits, dim=1)
                probs = torch.exp(log_probs)
                
                # Focal Term: (1 - probs)^gamma
                # Only applied to the target classes via element-wise multiplication
                gamma = 2.0
                focal_term = (1 - probs).pow(gamma)
                
                # Weighted Policy Loss with Focal Term
                # element_loss = - target * log(probs) * focal_term
                policy_loss_per_sample = -torch.sum(policy_target * log_probs * focal_term, dim=1)
                policy_loss = (policy_loss_per_sample * sample_weights.flatten()).mean()

                # Value: Weighted MSE
                value_loss_per_sample = (value_pred - value_target) ** 2
                value_loss = (value_loss_per_sample * sample_weights).mean()
                
                loss = policy_loss + value_loss

                # Metrics
                with torch.no_grad():
                    # Top-1 Accuracy
                    pred_move = torch.argmax(policy_logits, dim=1)
                    target_move = torch.argmax(policy_target, dim=1)
                    acc1 = (pred_move == target_move).float().mean()
                    
                    # Top-5 Accuracy
                    _, top5_moves = torch.topk(policy_logits, 5, dim=1)
                    acc5 = torch.sum(top5_moves == target_move.unsqueeze(1), dim=1).float().mean()
                    
                    # Value MAE
                    value_mae = torch.abs(value_pred - value_target).mean()
                    
                    # Policy Entropy
                    probs = torch.softmax(policy_logits, dim=1)
                    log_probs_metrics = torch.log_softmax(policy_logits, dim=1)
                    entropy = -torch.sum(probs * log_probs_metrics, dim=1).mean()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            # clip_grad_norm_ already returns the pre-clip norm. The old manual
            # per-parameter .item() loop forced hundreds of GPU synchronizations.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            metric_values = torch.stack([
                loss.detach().float(),
                policy_loss.detach().float(),
                value_loss.detach().float(),
                acc1.detach().float(),
                acc5.detach().float(),
                entropy.detach().float(),
                value_mae.detach().float(),
                total_norm.detach().float(),
            ]).cpu().tolist()
            (loss_value, policy_loss_value, value_loss_value, acc1_value,
             acc5_value, entropy_value, value_mae_value, grad_norm_value) = metric_values

            total_loss += loss_value
            total_policy_loss += policy_loss_value
            total_value_loss += value_loss_value
            total_acc1 += acc1_value
            total_acc5 += acc5_value
            total_entropy += entropy_value
            total_value_mae += value_mae_value
            batch_count += 1
            total_samples_seen += features.size(0)

            if batch_count % config.TRAIN_LOG_INTERVAL == 0 or batch_count == 1:
                pbar.set_postfix({
                    'loss': f"{loss_value:.3f}",
                    'p_loss': f"{policy_loss_value:.3f}",
                    'v_loss': f"{value_loss_value:.3f}",
                    'acc1': f"{acc1_value:.3f}",
                })
                writer.add_scalar('train/total_loss', loss_value, global_step)
                writer.add_scalar('train/policy_loss', policy_loss_value, global_step)
                writer.add_scalar('train/value_loss', value_loss_value, global_step)
                writer.add_scalar('train/accuracy_top1', acc1_value, global_step)
                writer.add_scalar('train/gradient_norm', grad_norm_value, global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

        # Step the scheduler at the end of each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        n_batches = max(1, batch_count)
        epoch_seconds = time.perf_counter() - epoch_started_at
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'accuracy_top1': total_acc1 / n_batches,
            'accuracy_top5': total_acc5 / n_batches,
            'policy_entropy': total_entropy / n_batches,
            'value_mae': total_value_mae / n_batches,
            'lr': current_lr,
            'duration_seconds': epoch_seconds,
            'samples_per_second': len(dataset) / max(epoch_seconds, 1e-9),
        }
        all_epoch_metrics.append(epoch_metrics)
        print(
            f"Epoch {epoch+1} complete | loss={epoch_metrics['loss']:.4f} "
            f"| {epoch_metrics['samples_per_second']:.1f} samples/s | lr={current_lr:.2e}"
        )

    writer.close()

    # 5. Save metrics for the parent loop/SwanLab process.
    training_seconds = time.perf_counter() - train_started_at
    train_metrics = dict(all_epoch_metrics[-1])
    train_metrics.update({
        'epochs': args.epochs,
        'generation': args.generation,
        'precision': precision,
        'seed': seed,
        'total_duration_seconds': training_seconds,
        'overall_samples_per_second': total_samples_seen / max(training_seconds, 1e-9),
        'epoch_metrics': all_epoch_metrics,
    })

    metrics_path = os.path.join(config.LOG_DIR, 'train_metrics.json')
    atomic_json_dump(train_metrics, metrics_path)
    print(f"Saved training metrics to {metrics_path}")

    # 6. Save a candidate checkpoint. Promotion to best.pth is owned by the
    # gating loop, so a failed candidate can never overwrite the incumbent.
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    model_to_save = getattr(model_to_save, '_orig_mod', model_to_save)
    checkpoint_payload = {
        'training_state_version': 2,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if use_grad_scaler else None,
        'generation': args.generation,
        'epochs_completed': args.epochs,
        'global_step': global_step,
        'precision': precision,
        'seed': seed,
        'metrics': train_metrics,
    }
    atomic_torch_save(checkpoint_payload, args.output)
    print(f"Saved candidate model to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_resume = config.CURRENT_MODEL_PTH if os.path.exists(config.CURRENT_MODEL_PTH) else config.INITIAL_MODEL_PTH
    parser.add_argument('--resume', type=str, default=default_resume)
    parser.add_argument('--data', type=str, default=None, help='Path to training data (file or directory)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the experiment run')
    parser.add_argument('--epochs', type=int, default=config.TRAIN_EPOCHS, help='Number of training epochs')
    parser.add_argument('--generation', type=int, default=-1)
    parser.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'], default=config.TRAIN_PRECISION)
    parser.add_argument('--output', type=str, default=os.path.join(config.CHECKPOINT_DIR, 'candidate.pth'))
    parser.add_argument(
        '--resume-optimizer',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Restore Adam moments when the checkpoint contains them.',
    )

    args = parser.parse_args()
    if args.resume and not os.path.exists(args.resume):
        parser.error(f"Resume checkpoint not found: {args.resume}")

    train(args)
