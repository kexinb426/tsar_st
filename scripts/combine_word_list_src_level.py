import json

file1 = "/clwork/kexin/tsar_st/data/source_with_difficulty.jsonl"   # first input
file2 = "/clwork/kexin/tsar_st/data/source_with_word_lists.jsonl"   # second input
out   = "/clwork/kexin/tsar_st/data/source_with_difficulty_word_lists.jsonl"   # output

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# Index by text_id from first file
by_id = {}
for obj in read_jsonl(file1):
    tid = obj["text_id"]
    by_id[tid] = obj

# Merge in second file (second wins on key conflicts)
for obj in read_jsonl(file2):
    tid = obj["text_id"]
    if tid in by_id:
        merged = {**by_id[tid], **obj}  # second overwrites duplicates
        by_id[tid] = merged
    else:
        by_id[tid] = obj

# Write merged output
with open(out, "w", encoding="utf-8") as f:
    for obj in by_id.values():
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")