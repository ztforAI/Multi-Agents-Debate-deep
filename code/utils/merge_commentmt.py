from pathlib import Path

# ===== 路径（已为你写死成你的实际路径）=====
zh_path = Path(r"D:\PycharmProjects\Multi-Agents-Debate\data\lexical.zh-en.zh")
en_path = Path(r"D:\PycharmProjects\Multi-Agents-Debate\data\lexical.zh-en.en")
out_path = Path(r"D:\PycharmProjects\Multi-Agents-Debate\data\lexical.zh-en.tsv")
# ===========================================

# 读取文件
zh_lines = zh_path.read_text(encoding="utf-8").splitlines()
en_lines = en_path.read_text(encoding="utf-8").splitlines()

# 去除首尾空白（不删行，保证一一对应）
zh_lines = [l.strip() for l in zh_lines]
en_lines = [l.strip() for l in en_lines]

# 行数一致性检查（非常重要）
if len(zh_lines) != len(en_lines):
    raise ValueError(
        f"Line count mismatch: zh={len(zh_lines)}, en={len(en_lines)}"
    )

# 写 TSV：source \t reference
with out_path.open("w", encoding="utf-8", newline="\n") as f:
    for zh, en in zip(zh_lines, en_lines):
        f.write(f"{zh}\t{en}\n")

print(f"✅ Merged {len(zh_lines)} lines")
print(f"📄 Output written to: {out_path}")
