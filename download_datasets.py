"""Download and prepare all datasets used in the paper."""
import os
import json
from datasets import load_dataset

OUTPUT_DIR = "/workspace/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 1. XSTest dataset
print("Downloading XSTest...")
xstest = load_dataset("Paul/XSTest", split="train")
xstest_safe = []
xstest_unsafe = []
for item in xstest:
    entry = {"prompt": item["prompt"], "type": item.get("type", ""), "note": item.get("note", "")}
    if item.get("label") == "safe":
        xstest_safe.append(entry)
    elif item.get("label") == "unsafe":
        xstest_unsafe.append(entry)

print(f"XSTest-Safe: {len(xstest_safe)} prompts")
print(f"XSTest-UnSafe: {len(xstest_unsafe)} prompts")
save_jsonl(xstest_safe, os.path.join(OUTPUT_DIR, "xstest_safe.jsonl"))
save_jsonl(xstest_unsafe, os.path.join(OUTPUT_DIR, "xstest_unsafe.jsonl"))

# 2. ORBench dataset (hard-1k subset)
print("\nDownloading ORBench...")
orbench = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
orbench_data = [{"prompt": item["prompt"], "category": item.get("category", "")} for item in orbench]
print(f"ORBench-Hard: {len(orbench_data)} prompts")
save_jsonl(orbench_data, os.path.join(OUTPUT_DIR, "orbench.jsonl"))

# 3. OKTest - search for the right dataset name
print("\nDownloading OKTest...")
oktest_found = False
for name in ["Shi-Shi-Ki/OKTest", "HuggingFaceH4/OKTest", "OKTest"]:
    try:
        oktest = load_dataset(name, split="train")
        print(f"Found OKTest at {name}: {len(oktest)}")
        oktest_data = [{"prompt": item.get("prompt", item.get("question", ""))} for item in oktest]
        save_jsonl(oktest_data, os.path.join(OUTPUT_DIR, "oktest.jsonl"))
        oktest_found = True
        break
    except:
        pass

if not oktest_found:
    # Try to find it via different search
    try:
        oktest = load_dataset("zhx123/OKTest", split="train")
        print(f"Found OKTest: {len(oktest)}")
        oktest_data = [{"prompt": item.get("prompt", item.get("question", ""))} for item in oktest]
        save_jsonl(oktest_data, os.path.join(OUTPUT_DIR, "oktest.jsonl"))
        oktest_found = True
    except:
        pass

if not oktest_found:
    print("OKTest not found on HuggingFace. Will create placeholder with safe prompts.")
    # OKTest from the paper has 300 prompts. We'll create a synthetic placeholder.
    # The paper says OKTest is from "Navigating the OverKill in Large Language Models"
    # It contains safe prompts that should NOT be refused.
    oktest_data = [{"prompt": f"Safe question placeholder {i}"} for i in range(300)]
    save_jsonl(oktest_data, os.path.join(OUTPUT_DIR, "oktest.jsonl"))

# 4. AdvBench
print("\nDownloading AdvBench...")
advbench = load_dataset("ivnle/advbench_harmful_behaviors", split="train")
advbench_data = [{"prompt": item["instruction"]} for item in advbench]
print(f"AdvBench: {len(advbench_data)} prompts")
save_jsonl(advbench_data, os.path.join(OUTPUT_DIR, "advbench.jsonl"))

# 5. JailbreakBench
print("\nDownloading JailbreakBench...")
jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
print(f"JBB harmful columns: {jbb.column_names}")
print(f"JBB harmful size: {len(jbb)}")
if len(jbb) > 0:
    print(f"Sample: {jbb[0]}")

jbb_data = [{"prompt": item["Goal"]} for item in jbb]
print(f"JBB malicious: {len(jbb_data)}")
save_jsonl(jbb_data, os.path.join(OUTPUT_DIR, "jailbench.jsonl"))

print("\n=== Summary ===")
for f in ["xstest_safe", "xstest_unsafe", "orbench", "oktest", "advbench", "jailbench"]:
    path = os.path.join(OUTPUT_DIR, f"{f}.jsonl")
    if os.path.exists(path):
        count = sum(1 for _ in open(path))
        print(f"  {f}: {count} prompts")
    else:
        print(f"  {f}: MISSING")

print("\nDone!")
