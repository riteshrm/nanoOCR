import random
import subprocess

# Get all image paths inside the archive
result = subprocess.run(
    ["tar", "-tzf", "mjsynth.tar.gz"],
    capture_output=True,
    text=True,
    check=True
)

files = [
    f for f in result.stdout.splitlines()
    if f.lower().endswith((".jpg", ".png"))
]

# Deterministic random selection
selected = random.Random(42).sample(files, 5)

# Save selected paths
with open("selected_files2.txt", "w") as f:
    f.write("\n".join(selected) + "\n")