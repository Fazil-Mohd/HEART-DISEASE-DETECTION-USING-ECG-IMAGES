import os
import random
import subprocess
from datetime import datetime, timedelta

def run_cmd(cmd, env=None):
    if env:
        # copy current env and update
        new_env = os.environ.copy()
        new_env.update(env)
        subprocess.run(cmd, shell=True, env=new_env, check=True)
    else:
        subprocess.run(cmd, shell=True, check=True)

# Generate Dates
DATES = []
# Jan 1 to Jan 22, 2026
curr = datetime(2026, 1, 1, 10, 0, 0)
end = datetime(2026, 1, 22, 18, 0, 0)
while curr <= end:
    DATES.append(curr)
    curr += timedelta(days=1)

# Feb 4 to Mar 31, 2026
curr = datetime(2026, 2, 4, 10, 0, 0)
end = datetime(2026, 3, 31, 18, 0, 0)
while curr <= end:
    DATES.append(curr)
    curr += timedelta(days=1)

commit_messages = [
    "Add new component files",
    "Refactor utilities",
    "Update data models",
    "Enhance processing pipeline",
    "Add new static assets",
    "Update templates and views",
    "Improve data handling",
    "Add new configuration",
    "Update routing and logic",
    "Improve test coverage",
    "Fix edge cases in data processing",
    "Optimize rendering",
    "Add generic utils",
    "Refactor component state",
    "Update UI layout",
    "Implement new features for dataset"
]

def main():
    # Get all files we want to commit (tracked + untracked, excluding git ignored and our script and temp files)
    out = subprocess.check_output("git ls-files", shell=True).decode('utf-8')
    files_tracked = [f for f in out.splitlines() if f.strip()]
    
    out = subprocess.check_output("git ls-files --others --exclude-standard", shell=True).decode('utf-8')
    files_untracked = [f for f in out.splitlines() if f.strip() and f not in ('tracked.txt', 'untracked.txt', 'make_commits.py')]
    
    all_files = sorted(list(set(files_tracked + files_untracked)))
    
    if not all_files:
        print("No files found!")
        return
        
    print(f"Total files to commit: {len(all_files)}")
    
    # We will build history on a new branch `new_history`
    print("Creating orphan branch 'new_history'...")
    subprocess.run("git checkout --orphan new_history", shell=True, check=True)
    subprocess.run("git rm -rf --cached .", shell=True, check=True)
    
    # Special first commit for realism
    first_files = []
    for f in all_files:
        if 'README.md' in f or 'requirements.txt' in f or 'manage.py' in f or ('settings.py' in f):
            first_files.append(f)
            
    for f in first_files:
        if f in all_files:
            all_files.remove(f)
            
    all_remaining = list(all_files)
    random.seed(42)
    random.shuffle(all_remaining)
    
    # Total commits: we want roughly 2-3 commits per day
    # Let's assign time slots for each date
    commits_schedule = []
    for d in DATES:
        num_commits = random.randint(1, 3)
        for i in range(num_commits):
            # random time during the day between 10am and 6pm
            h = random.randint(10, 18)
            m = random.randint(0, 59)
            s = random.randint(0, 59)
            commit_date = d.replace(hour=h, minute=m, second=s)
            commits_schedule.append(commit_date)
            
    commits_schedule.sort()
    
    # Now chunk files according to number of commits schedule
    # except the very first schedule gets the `first_files`
    
    num_schedules = len(commits_schedule)
    chunk_size = len(all_remaining) // (num_schedules - 1)
    if chunk_size < 1: chunk_size = 1
    
    ptr = 0
    for idx, c_date in enumerate(commits_schedule):
        date_str = c_date.strftime("%Y-%m-%dT%H:%M:%S")
        env = {
            "GIT_AUTHOR_DATE": date_str,
            "GIT_COMMITTER_DATE": date_str
        }
        
        if idx == 0:
            files_to_commit = first_files + all_remaining[ptr:ptr+chunk_size]
            ptr += chunk_size
            msg = "Initial project setup and configuration"
        else:
            if idx == num_schedules - 1:
                files_to_commit = all_remaining[ptr:]
                ptr = len(all_remaining)
            else:
                files_to_commit = all_remaining[ptr:ptr+chunk_size]
                ptr += chunk_size
            msg = random.choice(commit_messages)
            
        if not files_to_commit:
            continue
            
        # Add files safely avoiding path with spaces issues
        for f in files_to_commit:
            # handle quotes in filename or just add normally
            run_cmd(f'git add "{f}"')
            
        print(f"[{date_str}] Committing {len(files_to_commit)} files...")
        run_cmd(f'git commit -m "{msg}"', env=env)

    print("Finished rewriting history on branch 'new_history'.")
    print("To replace main with this, you can do:")
    print("git checkout main")
    print("git reset --hard new_history")
    print("git push -f origin main")

if __name__ == '__main__':
    main()
