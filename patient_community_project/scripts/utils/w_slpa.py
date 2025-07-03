import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Custom SLPA with weighted label propagation
def weighted_slpa(graph, t=20, r=0.1, plot_freq_dist=False, freq_dist_path=None):
    memory = {node: [node] for node in graph.nodes()}
    for t in range(t):
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        for listener in nodes:
            label_counts = defaultdict(float)
            for neighbor in graph.neighbors(listener):
                speaker_memory = memory[neighbor]
                label = random.choice(speaker_memory)
                weight = graph[listener][neighbor].get('weight', 0.0)
                label_counts[label] += weight
            if label_counts:
                selected_label = max(label_counts.items(), key=lambda x: x[1])[0]
                memory[listener].append(selected_label)
    # Post-processing
    communities = defaultdict(list)
    all_freqs = []  # For plotting frequency distribution
    for node, labels in memory.items():
        label_freq = defaultdict(int)
        for label in labels:
            label_freq[label] += 1
        total = len(labels)
        for label, count in label_freq.items():
            freq = count / total
            all_freqs.append(freq)
            if freq >= r:
                communities[label].append(node)
    print(all_freqs)
    if plot_freq_dist:
        plt.figure(figsize=(8, 4))
        plt.hist(all_freqs, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Label Frequencies in Node Memories')
        plt.xlabel('Frequency (count/total)')
        plt.ylabel('Number of (node, label) pairs')
        plt.axvline(r, color='red', linestyle='--', label=f'r = {r}')
        plt.legend()
        plt.tight_layout()
        if freq_dist_path:
            plt.savefig(freq_dist_path)
            print(f"Frequency distribution plot saved to {freq_dist_path}")
        else:
            plt.show()
        plt.close()
    return list(communities.values())