import re
import matplotlib.pyplot as plt

def extract_times(dump_file):
    pattern = r'\[info\]\s+\[(\d+)\]'
    times = []
    
    with open(dump_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                times.append(int(match.group(1)))
    
    return times

def plot_times(times, output_file='time_distribution.png', threshold=100):
    plt.figure(figsize=(12, 6))
    
    indices = list(range(len(times)))
    plt.plot(indices, times, marker='o', markersize=2, linewidth=1, label='Time')
    
    large_gaps = []
    for i in range(1, len(times)):
        gap = times[i] - times[i-1]
        if gap > threshold:
            large_gaps.append((i, times[i], gap))
    
    if large_gaps:
        gap_indices = [g[0] for g in large_gaps]
        gap_times = [g[1] for g in large_gaps]
        plt.scatter(gap_indices, gap_times, color='red', s=50, zorder=5, label='Large Gap')
        
        for idx, t, gap in large_gaps:
            plt.annotate(f'{t}\n(gap:{gap})', xy=(idx, t), xytext=(5, 10),
                        textcoords='offset points', fontsize=8, color='red')
    
    plt.xlabel('Record Index')
    plt.ylabel('Time')
    plt.title('Time Distribution from Dump File')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Chart saved to: {output_file}")
    
    if large_gaps:
        print(f"\nFound {len(large_gaps)} records with gap > {threshold}:")
        for idx, t, gap in large_gaps:
            print(f"  Index {idx}: time={t}, gap={gap}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dump_time_analysis.py <dump_file> [output_image] [threshold]")
        sys.exit(1)
    
    dump_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'time_distribution.png'
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    times = extract_times(dump_file)
    
    if not times:
        print("No time data found in dump file")
        sys.exit(1)
    
    print(f"Extracted {len(times)} time records")
    print(f"Min: {min(times)}, Max: {max(times)}, Avg: {sum(times)/len(times):.2f}")
    
    plot_times(times, output_file, threshold)