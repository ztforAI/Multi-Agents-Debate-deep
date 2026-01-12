# 简单测试脚本
import requests

API_KEY = "sk-3792687bb4804a1d8f97f6c61cbb17e3"  # ← 这里填入

response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 10
    }
)

print("状态码:", response.status_code)
if response.status_code == 200:
    print("✅ API Key 有效！")
else:
    print("❌ API Key 无效:", response.text)