'''
set -e
set -u

# Windows 路径在 bash 下建议用 /d/... 这种格式
MAD_PATH="/d/PycharmProjects/Multi-Agents-Debate"

python3 "$MAD_PATH/code/debate4tran.py" \
  -i "$MAD_PATH/data/CommonMT/input.example.txt" \
  -o "$MAD_PATH/data/CommonMT/output" \
  -lp zh-en \
  -k "sk-3792687bb4804a1d8f97f6c61cbb17e3"
'''
python code/debate4tran.py `
  -i data/CommonMT/input.example.txt `
  -o data/CommonMT/output `
  -lp zh-en `
  -k "sk-3792687bb4804a1d8f97f6c61cbb17e3"
