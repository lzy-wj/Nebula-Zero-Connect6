import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add path to core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def parse_game_length(moves_str):
    if not isinstance(moves_str, str):
        return 0
    # Moves are typically separated by ; or just parsed by length
    # Check format. Usually "M1;M2;..."
    # If empty or nan
    if moves_str == 'nan' or not moves_str:
        return 0
    # Try semicolon first
    if ';' in moves_str:
        return moves_str.count(';') + 1
    # Try comma
    if ',' in moves_str:
        return moves_str.count(',') + 1
    # Single move?
    if len(moves_str) > 1: return 1
    return 0

def analyze_file(file_path, label="Data"):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    print(f"\n{'='*20} Analyzing {label} {'='*20}")
    print(f"Source: {file_path}")

    # Read CSV
    # Headers: moves;winner;policy;bonus
    # Separator might be ; or | depending on version. Run loop uses | for eval, generate uses ;.
    # We'll try dynamic detection or fallback.
    
    # Try reading with comma separator first (most likely for gen files)
    # Also handle no header
    try:
        # 1. Try with header inference
        df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
        
        # Check if first row looks like data (not header)
        # If 'winner' column is NOT found, or if the columns look like data values
        if 'winner' not in df.columns or 'black' in str(df.columns) or 'white' in str(df.columns):
            # Reload with header=None
            df = pd.read_csv(file_path, sep=',', header=None, on_bad_lines='skip')
            # Assign columns based on count
            if len(df.columns) >= 2:
                # Standard format: moves, winner, policy, bonus
                df.columns = ['moves', 'winner', 'policy', 'bonus'][:len(df.columns)]
                
    except Exception as e:
        print(f"Error reading CSV with comma: {e}")
        return

    # Fallback to other separators if comma failed significantly (e.g. 1 column)
    if len(df.columns) < 2:
        try:
             df = pd.read_csv(file_path, sep=';', on_bad_lines='skip')
             if 'winner' not in df.columns:
                 df = pd.read_csv(file_path, sep='|', on_bad_lines='skip')
        except:
             pass

    total_games = len(df)
    print(f"Total Games: {total_games}")
    
    if total_games == 0:
        return

    # Normalize Winner Column
    # It might be 'black', 'white', or 1, -1
    if 'winner' in df.columns:
        # Convert to string and lower
        df['winner'] = df['winner'].astype(str).str.lower()
        
        def map_winner(val):
            if 'black' in val: return 1
            if 'white' in val: return -1
            if 'draw' in val: return 0
            try:
                f = float(val)
                return 1 if f > 0 else (-1 if f < 0 else 0)
            except:
                return 0 # Unknown
        
        df['winner'] = df['winner'].apply(map_winner)

    # 1. Win Distribution
    # winner: 1 (Black), -1 (White), 0 (Draw)
    black_wins = df[df['winner'] == 1]
    white_wins = df[df['winner'] == -1]
    draws = df[df['winner'] == 0]
    
    print(f"\n--- Results Distribution ---")
    print(f"Black Wins: {len(black_wins)} ({len(black_wins)/total_games:.2%})")
    print(f"White Wins: {len(white_wins)} ({len(white_wins)/total_games:.2%})")
    print(f"Draws:      {len(draws)} ({len(draws)/total_games:.2%})")
    
    # 2. Game Lengths
    # Apply length parsing
    # Use moves column (first column usually, or named 'moves')
    col_moves = 'moves' if 'moves' in df.columns else df.columns[0]
    
    lengths = df[col_moves].apply(parse_game_length)
    
    print(f"\n--- Game Length Stats (Moves) ---")
    print(f"Average: {lengths.mean():.2f}")
    print(f"Median:  {lengths.median():.2f}")
    print(f"Min:     {lengths.min()}")
    print(f"Max:     {lengths.max()}")
    
    # 3. Histogram buckets
    bins = [0, 20, 40, 60, 80, 100, 150, 200, 300]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-150', '151-200', '200+']
    # Use cut to bucket
    # Note: bins need to cover the range.
    cats = pd.cut(lengths, bins=bins + [999], labels=labels + ['300+'], right=True)
    counts = cats.value_counts(sort=False)
    
    print(f"\n--- Length Distribution ---")
    for label, count in counts.items():
        if count > 0:
            print(f"{label}: {count} ({count/total_games:.2%})")

    # 4. Win Rate by Length (Short vs Long)
    # Are short games dominated by Black?
    df['length'] = lengths
    
    short_games = df[df['length'] <= 40]
    long_games = df[df['length'] > 60]
    
    if len(short_games) > 0:
        sw = short_games[short_games['winner'] == -1]
        sb = short_games[short_games['winner'] == 1]
        print(f"\n--- Short Games (<=40 moves) ---")
        print(f"Count: {len(short_games)}")
        print(f"Black Win: {len(sb)/len(short_games):.2%}")
        print(f"White Win: {len(sw)/len(short_games):.2%}")

    if len(long_games) > 0:
        lw = long_games[long_games['winner'] == -1]
        lb = long_games[long_games['winner'] == 1]
        print(f"\n--- Long Games (>60 moves) ---")
        print(f"Count: {len(long_games)}")
        print(f"Black Win: {len(lb)/len(long_games):.2%}")
        print(f"White Win: {len(lw)/len(long_games):.2%}")

    # 5. Opening Diversity (First 4 moves)
    # Extract first few chars? Or full moves?
    # Format: aa;bb;cc...
    def get_opening(m_str, n=6):
        if not isinstance(m_str, str): return ""
        parts = m_str.split(';')
        return ";".join(parts[:n])

    openings = df[col_moves].apply(lambda x: get_opening(x, 5))
    unique_openings = openings.nunique()
    print(f"\n--- Diversity ---")
    print(f"Unique Openings (First 6 moves): {unique_openings}")
    print(f"Diversity Ratio: {unique_openings/total_games:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Connect6 Data")
    parser.add_argument('--generation', type=int, help="Generation ID to analyze (e.g. 50)")
    parser.add_argument('--buffer', action='store_true', help="Analyze the main replay buffer")
    parser.add_argument('--file', type=str, help="Specific file path to analyze")
    parser.add_argument('--new', action='store_true', help="Analyze the latest generation found")
    
    args = parser.parse_args()
    
    # Determine files to analyze
    files = []
    
    if args.file:
        files.append((args.file, "Custom File"))
        
    if args.generation:
        path = os.path.join(config.RAW_DATA_DIR, f'gen_{args.generation}.csv')
        files.append((path, f"Generation {args.generation}"))
        
    if args.buffer:
        path = os.path.join(config.BUFFER_DIR, 'replay_buffer.csv')
        files.append((path, "Replay Buffer"))
        
    if args.new:
        # 查找最新的 gen 文件
        gen_files = sorted([f for f in os.listdir(config.RAW_DATA_DIR) if f.startswith('gen_') and f.endswith('.csv')])
        if gen_files:
            # Sort by number: gen_X.csv
            try:
                gen_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest = gen_files[-1]
                path = os.path.join(config.RAW_DATA_DIR, latest)
                files.append((path, f"Latest Generation ({latest})"))
                
                # Also add the one before for comparison?
                if len(gen_files) > 1:
                    prev = gen_files[-2]
                    # files.append((os.path.join(config.RAW_DATA_DIR, prev), f"Previous Generation ({prev})"))
            except:
                pass

    if not files:
        print("Please specify --generation, --buffer, --new, or --file")
        return

    for path, label in files:
        analyze_file(path, label)

if __name__ == "__main__":
    main()
