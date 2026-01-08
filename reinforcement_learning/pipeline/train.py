import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import swanlab
import matplotlib.pyplot as plt
import io
from PIL import Image

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import C6TransNet
from core.connect6_game import Connect6Game
import config

import csv

# 安全的 SwanLab 日志函数（处理子进程情况）
def safe_swanlab_log(data, step=None):
    """包装 swanlab.log，在子进程中不会崩溃"""
    try:
        if step is not None:
            safe_swanlab_log(data, step=step)
        else:
            safe_swanlab_log(data)
    except RuntimeError:
        # swanlab.init 未调用（子进程中常见）
        pass
    except Exception as e:
        print(f"SwanLab log warning: {e}")

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
                count = moves_str.count(';') + 1
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
                    return int(m_str)
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
        
        # Reconstruct board at this state
        # Optimization: We don't need full replay if we just want one state.
        # But we need to place stones.
        # For simplicity/correctness: Replay up to move_idx.
        
        # TODO: Move this to C++ or optimize. For now, Python replay.
        board = np.zeros((19, 19), dtype=np.int8)
        
        # Replay
        # Move 0: Black (1 stone)
        # Move k>0: White/Black (2 stones) - Wait, moves list is coordinates.
        # Connect6Game logic:
        # First move: 1 stone. Subsequent: 2 stones.
        # The moves_str contains coordinates (0-360).
        # Wait, generate.py saves coordinates (0-360).
        
        # Let's look at generate.py:
        # game.moves.append(coord) -> coord is int 0-360
        
        # Replay Logic:
        current_player = 1 # Black
        
        # We need to replay ALL moves up to move_idx
        for i in range(move_idx):
            m = moves[i]
            r, c = m // 19, m % 19
            board[r, c] = current_player
            
            # Update player?
            # Connect6: 
            # i=0 (1st move): 1 stone. Next turn.
            # i=1 (2nd move): 1st stone of White.
            # i=2 (3rd move): 2nd stone of White. Next turn.
            # i=3 (4th move): 1st stone of Black.
            # ...
            
            # Turn logic:
            # Move 0: Black
            # Move 1,2: White
            # Move 3,4: Black
            # Move 5,6: White
            
            # Actually, let's just use the simple rule:
            # Stones on board count.
            stones = i + 1
            # Next player for move i+1?
            # If stones == 1, next is White.
            # If stones > 1: if (stones+1)//2 is odd -> White, else Black?
            # Let's use the logic from game.py:
            # rank = (n_stones + 1) // 2
            # if rank % 2 == 1: next = -1 (White) else 1 (Black)
            
            # But wait, we need the player who MADE the move to set the board.
            # Correct logic:
            # i=0: Black
            # i=1: White
            # i=2: White
            # i=3: Black
            # i=4: Black
            
            rank = (i + 1 + 1) // 2 # i is 0-indexed count of stones before this move
            # No, let's just track it.
            if i == 0:
                current_player = -1
            elif (i % 2) == 0: # i=2, 4, 6... (2nd stone of a pair, or start of new pair?)
                # i=1 (2nd stone overall, 1st of White): White. Next is White.
                # i=2 (3rd stone overall, 2nd of White): White. Next is Black.
                pass
            
            # Easier: Re-implement 'stones to place' logic or just use the formula
            # Move i was placed by:
            # 0 -> Black
            # 1,2 -> White
            # 3,4 -> Black
            # 5,6 -> White
            # Formula: if i == 0: Black. Else: ((i+1)//2) % 2 == 1 ? White : Black
        
        # Reset for the target state
        board.fill(0)
        current_player = 1
        for i in range(move_idx):
            m = moves[i]
            r, c = m // 19, m % 19
            
            # Who placed this stone?
            p = 1
            if i == 0: p = 1
            else:
                if ((i+1)//2) % 2 == 1: p = -1
                else: p = 1
            
            board[r, c] = p
        
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
                    k, v = item.split(':')
                    policy_target[int(k)] = float(v)
        else:
            # Should not happen unless log mismatch
            pass
            
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        print("Dataset is empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=4)
    
    # 2. Model
    model = C6TransNet(input_planes=17).to(device)
    
    # Load checkpoint
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
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
    # Separate LR for backbone?
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    # Cosine Annealing Scheduler: Decays from LR to MIN_LR over args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config.MIN_LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # SwanLab: Don't init here since run_loop.py already initializes it
    # Just log directly to the existing session
    # If called standalone, swanlab.log will fail gracefully
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    global_step = 0
    
    # 4. Training Loop
    model.train()
    
    # 收集每个 epoch 的指标，训练结束后保存到 JSON
    all_epoch_metrics = []
    for epoch in range(args.epochs):
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_acc1 = 0
        total_acc5 = 0
        total_entropy = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for features, policy_target, value_target, sample_weights in pbar:
            features = features.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device).unsqueeze(1) # (B, 1)
            sample_weights = sample_weights.to(device).unsqueeze(1) # (B, 1)
            
            # Forward
            with torch.cuda.amp.autocast():
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
                policy_loss = (policy_loss_per_sample * sample_weights.squeeze()).mean()
                
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

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Unscale before clipping/norm calc
            scaler.unscale_(optimizer)
            
            # Calculate Global Gradient Norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()
            total_entropy += entropy.item()
            batch_count += 1
            
            pbar.set_postfix({
                'loss': loss.item(), 
                'p_loss': policy_loss.item(), 
                'v_loss': value_loss.item(),
                'acc1': acc1.item()
            })
            
            # Log to TensorBoard and SwanLab
            writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Policy_Loss', policy_loss.item(), global_step)
            writer.add_scalar('Train/Value_Loss', value_loss.item(), global_step)
            writer.add_scalar('Train/Accuracy_Top1', acc1.item(), global_step)
            
            safe_swanlab_log({
                "Train/Total_Loss": loss.item(),
                "Train/Policy_Loss": policy_loss.item(),
                "Train/Value_Loss": value_loss.item(),
                "Train/Accuracy_Top1": acc1.item(),
                "Train/Accuracy_Top5": acc5.item(),
                "Train/Value_MAE": value_mae.item(),
                "Train/Policy_Entropy": entropy.item(),
                "Train/Global_Grad_Norm": total_norm,
                "Train/LR": optimizer.param_groups[0]['lr']
            }, step=global_step)
            
            global_step += 1
        
        # Step the scheduler at the end of each epoch
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        print(f"Epoch {epoch+1} Complete. LR decayed to {current_lr:.2e}")
            
    writer.close()
    
    # 5. 保存训练指标到 JSON（供 run_loop 统一上传到 SwanLab）
    import json
    
    # 计算最后一个 epoch 的平均指标
    n_batches = batch_count if batch_count > 0 else 1
    train_metrics = {
        'loss': total_loss / n_batches,
        'policy_loss': total_policy_loss / n_batches,
        'value_loss': total_value_loss / n_batches,
        'accuracy_top1': total_acc1 / n_batches,
        'accuracy_top5': total_acc5 / n_batches,
        'policy_entropy': total_entropy / n_batches,
        'epochs': args.epochs,
    }
    
    metrics_path = os.path.join(config.LOG_DIR, 'train_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")
            
    # 6. 保存模型
    out_path = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=config.INITIAL_MODEL_PATH.replace('.engine', '.pth')) 
    parser.add_argument('--data', type=str, default=None, help='Path to training data (file or directory)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the experiment run')
    parser.add_argument('--epochs', type=int, default=config.TRAIN_EPOCHS, help='Number of training epochs')
    
    # Note: Initial path is engine, we need PTH for training. 
    # Assuming the user provides a valid PTH or we find one.
    # Let's check if best.pth exists in checkpoints, use that.
    
    args = parser.parse_args()
    
    # 优先使用 checkpoints/best.pth（如果存在）
    best_pth = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
    if os.path.exists(best_pth):
        args.resume = best_pth
    elif not os.path.exists(args.resume):
        print(f"Warning: Resume path not found: {args.resume}")
        print(f"Please provide a valid checkpoint via --resume")
        
    train(args)
