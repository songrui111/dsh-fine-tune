import io, json, os, re, statistics

SYS = "[\u7cfb\u7edf\u63d0\u793a]"
USER = "[\u7528\u6237]"
TOOL = "[\u5de5\u5177\u8fd4\u56de]"
MARK = re.compile("(" + "|".join(re.escape(x) for x in (SYS, USER, TOOL)) + ")")

def condense(prompt, keep_tools=2, tool_chars=250, user_chars=300):
    parts = MARK.split(prompt)
    blocks = [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(1, len(parts), 2)]
    users = [b for k, b in blocks if k == USER]
    tools = [b for k, b in blocks if k == TOOL]
    out = []
    if users: out.append(USER + "\n" + users[-1].strip()[:user_chars])
    for t in tools[-keep_tools:]:
        body = t.strip()
        out.append(TOOL + "\n" + (body[:tool_chars] + ("\u2026" if len(body) > tool_chars else "")))
    return "\n\n".join(out) if out else prompt[-300:]

os.environ["HF_HOME"] = r"C:\Users\86156\Documents\deepseek-harness\default-workspace\.tmp-torch\hf"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

src = os.path.join(os.path.expanduser("~"), ".dsh", "fine-tune", "datasets", "m0-act2", "train.jsonl")
toks = []
with io.open(src, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 300: break
        row = json.loads(line)
        p = condense(row["messages"][0]["content"])
        l = row["messages"][-1]["content"]
        toks.append(len(tok(p).input_ids) + len(tok(l).input_ids))
toks.sort()
print("real token counts after condensing (300 samples):")
print("  min/median/p90/max:", toks[0], int(statistics.median(toks)), toks[int(len(toks)*0.9)], toks[-1])
for limit in (512, 1024, 1536):
    n = sum(1 for t in toks if t <= limit)
    print(f"  fits in {limit:4}: {n:3}/{len(toks)} ({n*100//len(toks)}%)")
