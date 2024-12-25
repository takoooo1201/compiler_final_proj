import subprocess
import glob

for lsp_file in glob.glob("public_test_data/*.lsp"):
    print(f"Running {lsp_file}...")
    result = subprocess.run(["python", "test.py", lsp_file], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)