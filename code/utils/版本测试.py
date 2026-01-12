import openai
import tiktoken
import langcodes
import language_data
import backoff

print(f"--- 环境检查 ---")
print(f"OpenAI 版本: {openai.__version__} (应 > 1.0.0)")
print(f"tiktoken 版本: {tiktoken.__version__} (应 > 0.5.0)")
try:
    # 测试翻译逻辑的核心库
    display_name = langcodes.Language.get('zh').display_name()
    print(f"语言解析测试: 'zh' -> {display_name} (应显示 Chinese)")
except Exception as e:
    print(f"语言库版本冲突: {e}")