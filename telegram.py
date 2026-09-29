import os
import requests
import config
from heros import ingest_by_filename, ingest_description


def _get_file_path(file_id: str) -> str:
    response = requests.get(
        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getFile",
        params={"file_id": file_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise ValueError(f"Telegram getFile failed: {data.get('description')}")
    return data["result"]["file_path"]


def _download_file(file_path: str) -> bytes:
    response = requests.get(
        f"https://api.telegram.org/file/bot{config.TELEGRAM_BOT_TOKEN}/{file_path}",
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def get_documents_from_updates() -> list:
    response = requests.get(
        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates",
        params={"limit": 100},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise ValueError(f"Telegram getUpdates failed: {data.get('description')}")

    docs = []
    for update in data["result"]:
        msg = update.get("message") or update.get("channel_post", {})
        document = msg.get("document")
        if document:
            docs.append({
                "file_id": document["file_id"],
                "filename": document.get("file_name", "unknown"),
                "mime_type": document.get("mime_type"),
                "file_size": document.get("file_size"),
                "date": msg.get("date"),
            })
    return docs


def ingest_from_telegram(conn, file_id: str, filename: str, description: str = None) -> dict:
    # 1. get file path from telegram
    file_path = _get_file_path(file_id)

    # 2. download file content
    content = _download_file(file_path)

    # 3. save temporarily to documents folder
    os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
    temp_path = os.path.join(config.DOCUMENTS_DIR, filename)
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # 4. reuse existing ingest logic
        result = ingest_by_filename(conn, filename, description)
        res = ingest_description(conn, filename)
    finally:
        # 5. delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result
