import time
import pandas as pd
import fireducks as fd  # Note: This is a mock import for demonstration
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Configuration
DATA_SIZE = 1000000  # 1 million rows
TEST_SIZE = 0.2
NUM_FEATURES = 20
CATEGORICAL_FEATURES = 5
NUM_CATEGORIES = 10

# Generate synthetic data
def generate_data(size):
    print(f"Generating synthetic data with {size} rows...")
    np.random.seed(42)
    
    # Numerical features
    data = {
        f'num_{i}': np.random.normal(0, 1, size) for i in range(NUM_FEATURES)
    }
    
    # Categorical features
    for i in range(CATEGORICAL_FEATURES):
        data[f'cat_{i}'] = np.random.randint(0, NUM_CATEGORIES, size)
    
    # Target variable
    data['target'] = np.random.randint(0, 2, size)
    
    return data

# Benchmark functions
def benchmark_operation(name, pandas_func, fireducks_func, data, repeat=3):
    print(f"\nBenchmarking {name}...")
    
    # Pandas benchmark
    pd_times = []
    for _ in range(repeat):
        df_pd = pd.DataFrame(data.copy())
        start = time.time()
        pandas_func(df_pd)
        pd_times.append(time.time() - start)
    pd_avg = np.mean(pd_times)
    
    # FireDucks benchmark
    fd_times = []
    for _ in range(repeat):
        df_fd = fd.DataFrame(data.copy())  # Mock FireDucks DataFrame
        start = time.time()
        fireducks_func(df_fd)
        fd_times.append(time.time() - start)
    fd_avg = np.mean(fd_times)
    
    speedup = pd_avg / fd_avg if fd_avg > 0 else 0
    
    print(f"Pandas: {pd_avg:.4f}s | FireDucks: {fd_avg:.4f}s | Speedup: {speedup:.1f}x")
    
    return {
        'operation': name,
        'pandas_time': pd_avg,
        'fireducks_time': fd_avg,
        'speedup': speedup
    }

# Operation implementations
def pd_load(data):
    return pd.DataFrame(data)

def fd_load(data):
    return fd.DataFrame(data)

def pd_clean(df):
    return df.dropna()

def fd_clean(df):
    return df.dropna()

def pd_feature_eng(df):
    df = pd.get_dummies(df, columns=[f'cat_{i}' for i in range(CATEGORICAL_FEATURES)])
    df['new_feature'] = df['num_0'] * df['num_1']
    return df

def fd_feature_eng(df):
    df = df.get_dummies(columns=[f'cat_{i}' for i in range(CATEGORICAL_FEATURES)])
    df['new_feature'] = df['num_0'] * df['num_1']
    return df

def pd_train(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    return model

def fd_train(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    return model

def run_benchmarks():
    # Generate data once
    data = generate_data(DATA_SIZE)
    
    # List of benchmarks to run
    benchmarks = [
        ('Data Loading', pd_load, fd_load),
        ('Data Cleaning', pd_clean, fd_clean),
        ('Feature Engineering', pd_feature_eng, fd_feature_eng),
        ('Model Training', pd_train, fd_train)
    ]
    
    results = []
    for name, pd_func, fd_func in benchmarks:
        results.append(benchmark_operation(name, pd_func, fd_func, data))
    
    return results

def visualize_results(results):
    operations = [r['operation'] for r in results]
    pd_times = [r['pandas_time'] for r in results]
    fd_times = [r['fireducks_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Execution time comparison
    x = range(len(operations))
    width = 0.35
    ax1.bar(x, pd_times, width, label='Pandas')
    ax1.bar([p + width for p in x], fd_times, width, label='FireDucks')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks([p + width/2 for p in x])
    ax1.set_xticklabels(operations)
    ax1.legend()
    
    # Speedup comparison
    ax2.bar(x, speedups)
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('FireDucks Speedup Over Pandas')
    ax2.set_xticks(x)
    ax2.set_xticklabels(operations)
    ax2.axhline(1, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()
    
    # Print table
    print("\nBenchmark Results Summary:")
    print(f"{'Operation':<20} | {'Pandas Time (s)':>15} | {'FireDucks Time (s)':>18} | {'Speedup':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['operation']:<20} | {r['pandas_time']:15.4f} | {r['fireducks_time']:18.4f} | {r['speedup']:10.1f}x")

if __name__ == "__main__":
    print("Starting FireDucks vs Pandas Benchmarking")
    results = run_benchmarks()
    visualize_results(results)