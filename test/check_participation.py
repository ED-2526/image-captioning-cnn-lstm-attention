import subprocess

def get_commit_counts():
    result = subprocess.run(['git', 'shortlog', '-s', '-n'], capture_output=True, text=True, timeout=30)
    lines = result.stdout.strip().split('\n')
    commit_counts = {}
    for line in lines:
        count, name = line.strip().split('\t')
        commit_counts[name] = int(count)
    return commit_counts

def calculate_percentages(commit_counts):
    total_commits = sum(commit_counts.values())
    percentages = {name: (count / total_commits) * 100 for name, count in commit_counts.items()}
    return percentages

commit_counts = get_commit_counts()
percentages = calculate_percentages(commit_counts)

print("Commit Counts:")
for name, count in commit_counts.items():
    print(f"{name}: {count}")

print("\nPercentages:")
for name, percentage in percentages.items():
    print(f"{name}: {percentage:.2f}%")