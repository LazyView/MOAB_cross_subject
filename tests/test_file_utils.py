"""
Test file utilities to debug subject detection.
"""

from utils.file_utils import get_all_subjects, find_subject_files, parse_filename
from pathlib import Path

print("Testing file utilities...")
print("=" * 60)

# Test 1: Get all subjects
print("\n1. Getting all subjects:")
subjects = get_all_subjects("data/")
print(f"   Found {len(subjects)} subjects: {subjects}")

# Test 2: Parse a known file
print("\n2. Testing filename parsing:")
test_file = Path("data/01_12_2020/1z01122020lh1.vhdr")
if test_file.exists():
    file_info = parse_filename(test_file)
    if file_info:
        print(f"   Parsed successfully:")
        print(f"      Subject ID: {file_info.subject_id}")
        print(f"      Gender: {file_info.gender}")
        print(f"      Date: {file_info.date}")
        print(f"      Task: {file_info.task}")
        print(f"      Run: {file_info.run}")
    else:
        print(f"   Failed to parse: {test_file.name}")
else:
    print(f"   File not found: {test_file}")

# Test 3: Find files for subject 1
if subjects:
    print(f"\n3. Finding files for subject '{subjects[0]}':")
    files = find_subject_files("data/", subjects[0])
    print(f"   Left hand files: {len(files['lh'])}")
    for f in files['lh']:
        print(f"      {f.file_path.name}")
    print(f"   Right hand files: {len(files['rh'])}")
    for f in files['rh']:
        print(f"      {f.file_path.name}")
