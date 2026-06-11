import argparse
import json
import time
import hmac
import hashlib
import base64
import requests

def send_dingtalk_notification(webhook, secret, title, text):
    """
    发送钉钉通知
    :param webhook: 钉钉机器人Webhook地址
    :param secret: 钉钉机器人加签密钥
    :param title: 通知标题
    :param text: 通知内容（Markdown格式）
    """
    timestamp = str(round(time.time() * 1000))
    secret_enc = secret.encode('utf-8')
    string_to_sign = f"{timestamp}\n{secret}"
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = base64.b64encode(hmac_code).decode('utf-8')

    url = f"{webhook}&timestamp={timestamp}&sign={sign}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": text
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"发送钉钉通知失败: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Notification Sender")
    parser.add_argument("--dingtalk-webhook", required=True)
    parser.add_argument("--dingtalk-secret", default="")
    parser.add_argument("--project", required=True)
    parser.add_argument("--pr-number", required=True)
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--critical-count", required=True, type=int)
    parser.add_argument("--high-count", required=True, type=int)
    parser.add_argument("--medium-count", required=True, type=int)
    parser.add_argument("--low-count", required=True, type=int)
    args = parser.parse_args()

    status_emoji = "✅" if args.status == "success" else "❌"
    title = f"{status_emoji} AI代码审查结果 - {args.project}"
    
    text = f"""# {title}

**PR编号**: #{args.pr_number}
**PR链接**: [点击查看]({args.pr_url})
**审查结果**: {args.status}

## 📊 问题统计
| 严重程度 | 数量 |
|----------|------|
| 🔴 高危 | {args.critical_count} |
| 🟠 严重 | {args.high_count} |
| 🟡 中等 | {args.medium_count} |
| 🟢 轻微 | {args.low_count} |
"""

    if args.dingtalk_secret:
        send_dingtalk_notification(args.dingtalk_webhook, args.dingtalk_secret, title, text)
    else:
        # 未开启加签时直接发送
        response = requests.post(
            args.dingtalk_webhook,
            headers={"Content-Type": "application/json"},
            json={
                "msgtype": "markdown",
                "markdown": {"title": title, "text": text}
            }
        )
        if response.status_code != 200:
            print(f"发送钉钉通知失败: {response.text}")

if __name__ == "__main__":
    main()