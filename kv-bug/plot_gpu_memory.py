import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bisect_results.csv')

run_cols = [f'gpu_mem_run{i}_mb' for i in range(1, 11)]

plt.figure(figsize=(10, 6))

for idx, row in df.iterrows():
    values = row[run_cols]
    valid = values.dropna()
    if not valid.empty:
        x = [int(col.split('run')[1].split('_')[0]) for col in valid.index]
        # Use short_hash or index as label
        label = row.get('short_hash', f'Row {idx}')
        plt.plot(x, valid.values, marker='o', label=label)

plt.xlabel('Run Number')
plt.ylabel('GPU Memory (MB)')
plt.title('GPU Memory Usage Across Runs')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('gpu_memory_plot.png', dpi=150, bbox_inches='tight')
plt.show()
