import subprocess

# Define the list of scripts to run in order
scripts = [
    # "scripts/build_graph.py",
    "scripts/run_leiden.py",
    "scripts/run_w_slpa.py",
    "scripts/run_reverse_hybrid.py",

    # "scripts/run_slpa.py",
    # "scripts/run_hybrid.py",
]

# Run each script sequentially and print progress
for script in scripts:
    print(f"\n=== Running {script} ===")
    try:
        subprocess.run(["python", f"{script}"], check=True)
        print(f"✅ Completed {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run {script}: {e}")
        break

