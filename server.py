#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ИИ-ассистент для написания контента в блог.

Запуск:  python3 server.py
Открыть: http://127.0.0.1:8000

Нейросеть (по умолчанию бесплатно, локально):
  - Ollama (http://127.0.0.1:11434), модель выбирается автоматически.
  Если Ollama недоступна, используются API-ключи из окружения:
  - OPENAI_API_KEY  -> OpenAI
  - ANTHROPIC_API_KEY -> Anthropic Claude
"""

import csv
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_CSV = os.path.join(BASE_DIR, "ца_примеры_тон.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "сгенерированные_посты.csv")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")

MAX_HISTORY = 12

FORMAT_FOOTER = ("Отвечай строго в формате:\n"
                 "ЗАГОЛОВОК: <заголовок поста>\nТЕКСТ ПОСТА:\n<текст поста>")


def extract_pdf_text(raw):
    """Извлекает текст из PDF-файла (байты) через pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("Библиотека pypdf не установлена: pip install pypdf")
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as e:
        raise RuntimeError("Не удалось открыть PDF: %s" % e)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(
            "В PDF не найден текст. Возможно, это сканированный документ (изображения).")
    return text


def detect_delimiter(path):
    with open(path, "rb") as f:
        first = f.readline().decode("utf-8-sig", "replace")
    sem, com = first.count(";"), first.count(",")
    if sem >= com and sem > 0:
        return ";"
    return ","


def read_settings():
    """Читает строки из ца_примеры_тон.csv: ЦА, примеры текстов, тон контента."""
    if not os.path.exists(SETTINGS_CSV):
        return []
    delimiter = detect_delimiter(SETTINGS_CSV)
    rows = []
    with open(SETTINGS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            rows.append({
                "ца": (row.get("ЦА") or "").strip(),
                "примеры": (row.get("Примеры текстов") or "").strip(),
                "тон": (row.get("Тон контента") or "").strip(),
            })
    return rows


def append_post(title, text):
    """Дописывает пост (заголовок, текст) в сгенерированные_посты.csv."""
    new = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    encoding = "utf-8-sig" if new else "utf-8"
    with open(OUTPUT_CSV, "a", encoding=encoding, newline="") as f:
        w = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        if new:
            w.writerow(["Заголовок", "Текст поста"])
        w.writerow([title, text])


def handle_save_settings(request):
    """Сохраняет настройки (ЦА, примеры текстов, тон контента) в ца_примеры_тон.csv."""
    delimiter = (request.get("delimiter") or ";")
    if delimiter not in (";", ","):
        delimiter = ";"
    rows = request.get("rows") or []
    with open(SETTINGS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["ЦА", "Примеры текстов", "Тон контента"])
        for r in rows:
            w.writerow([(r.get("ца") or ""), (r.get("примеры") or ""), (r.get("тон") or "")])
    return {"ok": True, "rows": len(rows)}


def ollama_models():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            data = json.load(r)
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def _ollama_reachable():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2) as r:
            json.load(r)
        return True
    except Exception:
        return False


def ensure_ollama():
    """Проверяет Ollama, при необходимости запускает её и возвращает модели."""
    if _ollama_reachable():
        return {"running": True, "started": False, "models": ollama_models()}
    try:
        env = dict(os.environ)
        env["OLLAMA_ORIGINS"] = "*"
        subprocess.Popen(["ollama", "serve"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except FileNotFoundError:
        raise RuntimeError("Ollama не найден. Установите Ollama: https://ollama.com")
    for _ in range(30):
        time.sleep(1)
        if _ollama_reachable():
            return {"running": True, "started": True, "models": ollama_models()}
    raise RuntimeError("Не удалось запустить Ollama. Установите её и выполните `ollama pull qwen2.5:7b`.")


def pick_model(models):
    pref = OLLAMA_MODEL
    if pref in models:
        return pref
    if pref and pref + ":latest" in models:
        return pref + ":latest"
    for name in ("qwen2.5:7b", "qwen2.5:3b", "llama3.2", "llama3.1", "mistral", "gemma2", "llava"):
        if name in models:
            return name
    return models[0] if models else None


def _post(url, payload, headers, timeout=600):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = (e.read().decode("utf-8", "replace") or "").strip()[:200]
            except Exception:
                pass
        message = "Ошибка API (HTTP %s)" % e.code if isinstance(e, urllib.error.HTTPError) else ("Ошибка сети: %s" % e.reason)
        raise RuntimeError(message + (": " + detail if detail else ""))


def _get(url, headers, timeout=30):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        if isinstance(e, urllib.error.HTTPError):
            raise
        raise RuntimeError("Ошибка сети при обращении к провайдеру: %s" % e.reason)


def check_api_key(request):
    """Проверяет API-ключ внешнего ИИ лёгким запросом к провайдеру."""
    provider = (request.get("провайдер") or "").lower()
    api_key = (request.get("ключ") or "").strip()
    if not api_key:
        raise RuntimeError("Не указан API-ключ.")
    try:
        if "openai" in provider:
            data = _get("https://api.openai.com/v1/models",
                        {"Authorization": "Bearer " + api_key})
            name = "OpenAI"
        elif "deepseek" in provider:
            data = _get("https://api.deepseek.com/v1/models",
                        {"Authorization": "Bearer " + api_key})
            name = "DeepSeek"
        elif "proxyapi" in provider or "proxy" in provider:
            data = _get("https://api.proxyapi.ru/openai/v1/models",
                        {"Authorization": "Bearer " + api_key})
            name = "ProxyAPI"
        elif "anthropic" in provider or "claude" in provider:
            data = _get("https://api.anthropic.com/v1/models",
                        {"x-api-key": api_key, "anthropic-version": "2023-06-01"})
            name = "Anthropic Claude"
        else:
            raise RuntimeError("Неизвестный провайдер: " + (provider or ""))
        models = [m.get("id", "") for m in data.get("data", []) if m.get("id")][:300]
        return {"ok": True, "провайдер": name, "моделей": len(models), "модели": models}
    except urllib.error.HTTPError as e:
        name = {"openai": "OpenAI", "deepseek": "DeepSeek", "proxyapi": "ProxyAPI",
                "proxy": "ProxyAPI", "anthropic": "Anthropic", "claude": "Anthropic"}.get(provider, provider)
        if e.code == 401:
            raise RuntimeError(
                "%s отклонил API-ключ (HTTP 401). Проверьте, что ключ скопирован целиком, "
                "не просрочен и оформлен на нужного провайдера." % name)
        raise RuntimeError("Ошибка провайдера %s: HTTP %d" % (name, e.code))


def ollama_chat(messages, model):
    data = _post(OLLAMA_URL + "/api/chat",
                 {"model": model, "messages": messages, "stream": False},
                 {"Content-Type": "application/json"})
    return data.get("message", {}).get("content", "")


def openai_chat(messages, api_key=None, model=None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Не указан API-ключ OpenAI.")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    data = _post("https://api.openai.com/v1/chat/completions",
                 {"model": model, "messages": messages},
                 {"Content-Type": "application/json",
                  "Authorization": "Bearer " + key})
    return data["choices"][0]["message"]["content"]


def anthropic_chat(messages, api_key=None, model=None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Не указан API-ключ Anthropic.")
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"]
    data = _post("https://api.anthropic.com/v1/messages",
                 {"model": model,
                  "max_tokens": 2000, "system": system, "messages": msgs},
                 {"Content-Type": "application/json",
                  "x-api-key": key,
                  "anthropic-version": "2023-06-01"})
    return "".join(b.get("text", "") for b in data.get("content", []))


def deepseek_chat(messages, api_key=None, model=None):
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Не указан API-ключ DeepSeek.")
    model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    data = _post("https://api.deepseek.com/v1/chat/completions",
                 {"model": model, "messages": messages},
                 {"Content-Type": "application/json",
                  "Authorization": "Bearer " + key})
    return data["choices"][0]["message"]["content"]


def proxyapi_chat(messages, api_key=None, model=None):
    key = api_key or os.environ.get("PROXYAPI_API_KEY")
    if not key:
        raise RuntimeError("Не указан API-ключ ProxyAPI.")
    model = model or os.environ.get("PROXYAPI_MODEL", "gpt-4o-mini")
    data = _post("https://api.proxyapi.ru/openai/v1/chat/completions",
                 {"model": model, "messages": messages},
                 {"Content-Type": "application/json",
                  "Authorization": "Bearer " + key})
    return data["choices"][0]["message"]["content"]


def ask_external(provider, api_key, model, messages):
    p = (provider or "").lower()
    if "openai" in p:
        return "OpenAI", openai_chat(messages, api_key, model)
    if "anthropic" in p or "claude" in p:
        return "Anthropic Claude", anthropic_chat(messages, api_key, model)
    if "deepseek" in p:
        return "DeepSeek", deepseek_chat(messages, api_key, model)
    if "proxyapi" in p or "proxy" in p:
        return "ProxyAPI", proxyapi_chat(messages, api_key, model)
    raise RuntimeError("Неизвестный внешний ИИ: " + (provider or ""))


def provider_info():
    models = ollama_models()
    if models:
        return {"провайдер": "Ollama (локально, бесплатно)", "модель": pick_model(models),
                "модели": models}
    if os.environ.get("OPENAI_API_KEY"):
        return {"провайдер": "OpenAI", "модель": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "модели": []}
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"провайдер": "Anthropic Claude",
                "модель": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                "модели": []}
    return {"провайдер": "не выбран", "модель": "", "модели": []}


def ask_ai(messages):
    models = ollama_models()
    if models:
        model = pick_model(models)
        return "Ollama (локально, бесплатно)", model, ollama_chat(messages, model)
    if os.environ.get("OPENAI_API_KEY"):
        return "OpenAI", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), openai_chat(messages)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "Anthropic Claude", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"), anthropic_chat(messages)
    raise RuntimeError(
        "Нейросеть не найдена. Запустите Ollama (ollama serve) и установите модель "
        "`ollama pull qwen2.5:7b`, либо задайте OPENAI_API_KEY или ANTHROPIC_API_KEY.")


def build_system(role, ца, тон, примеры):
    parts = []
    if role:
        parts.append(f"Твоя роль: {role}.")
    parts.append("Ты — профессиональный контент-менеджер, пишешь посты для блога.")
    if ца:
        parts.append(f"Целевая аудитория: {ца}.")
    if тон:
        parts.append(f"Тон контента: {тон}.")
    if примеры:
        parts.append(f"Ориентируйся на стиль следующих примеров текстов:\n{примеры}")
    parts.append(FORMAT_FOOTER)
    return "\n".join(parts)


def build_system_from_request(request):
    промпт = (request.get("промпт") or "").strip()
    if промпт:
        return промпт + "\n\n" + FORMAT_FOOTER
    return build_system((request.get("роль") or "").strip(),
                        (request.get("ца") or "").strip(),
                        (request.get("тон") or "").strip(),
                        (request.get("примеры") or "").strip())


def parse_post(text):
    text = text.strip()
    if "ТЕКСТ ПОСТА:" in text:
        head, body = text.split("ТЕКСТ ПОСТА:", 1)
        title = ""
        for line in head.splitlines():
            s = line.strip()
            if s.upper().startswith("ЗАГОЛОВОК:"):
                title = s[len("ЗАГОЛОВОК:"):].strip().strip("\"'«»")
                break
        if not title:
            for line in head.splitlines():
                s = line.strip().lstrip("*-№. ")
                if s:
                    title = s
                    break
        return title, body.strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.upper().startswith("ЗАГОЛОВОК:"):
            title = s[len("ЗАГОЛОВОК:"):].strip().strip("\"'«»")
            return title, "\n".join(lines[i + 1:]).strip()
    if len(lines) > 1 and lines[0].strip():
        return lines[0].strip().lstrip("*-№. "), "\n".join(lines[1:]).strip()
    return "", text


def _ask(request, messages):
    """Выбирает нейросеть: внешний ИИ (провайдер+ключ) или стандартная (Ollama/ключи окружения)."""
    provider = (request.get("провайдер") or "").strip()
    api_key = (request.get("ключ") or "").strip()
    model = (request.get("модель") or "").strip()
    if provider:
        provider_name, answer = ask_external(provider, api_key, model, messages)
        return provider_name, model or "-", answer
    return ask_ai(messages)


def handle_apply_role(request):
    """Отправляет выбранную роль как системный промпт и возвращает приветствие нейросети."""
    messages = [
        {"role": "system", "content": build_system_from_request(request)},
        {"role": "user", "content":
            "Представься в своей роли и сообщи, что готов к работе. "
            "Ответь одним-двумя короткими предложениями. Без формата ЗАГОЛОВОК/ТЕКСТ ПОСТА."},
    ]
    provider, model, answer = _ask(request, messages)
    return {"ответ": answer.strip(), "провайдер": provider, "модель": model}


def handle_chat(request):
    role = (request.get("роль") or "").strip()
    ца = (request.get("ца") or "").strip()
    тон = (request.get("тон") or "").strip()
    примеры = (request.get("примеры") or "").strip()
    промпт = (request.get("промпт") or "").strip()
    history = request.get("история") or []
    text = (request.get("сообщение") or "").strip()
    if not text:
        raise ValueError("Сообщение не должно быть пустым.")

    system = (промпт + "\n\n" + FORMAT_FOOTER) if промпт else build_system(role, ца, тон, примеры)
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m.get("role"), "content": m.get("content")}
                 for m in history[-MAX_HISTORY:] if m.get("role") in ("user", "assistant") and m.get("content")]
    messages.append({"role": "user", "content": text})

    provider, model, answer = _ask(request, messages)
    title, body = parse_post(answer)
    append_post(title, body)
    return {"ответ": answer, "заголовок": title, "провайдер": provider,
            "модель": model, "сохранено": True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 5_000_000:
            raise ValueError("Некорректный размер запроса.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            path = os.path.join(BASE_DIR, "index.html")
            if not os.path.exists(path):
                return self.send_error(404)
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/settings":
            if os.path.exists(SETTINGS_CSV):
                self._send_json({"delimiter": detect_delimiter(SETTINGS_CSV), "rows": read_settings()})
            else:
                self._send_json({"delimiter": ";", "rows": []})
        elif self.path == "/api/status":
            self._send_json(provider_info())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/extract-pdf":
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > 20_000_000:
                    raise ValueError("Некорректный размер PDF-файла.")
                raw = self.rfile.read(length)
                text = extract_pdf_text(raw)
                self._send_json({"текст": text, "символов": len(text)})
                return
            except Exception as e:
                self._send_json({"ошибка": str(e)}, code=500)
                return
        if self.path == "/api/chat":
            try:
                self._send_json(handle_chat(self._read_json()))
                return
            except Exception as e:
                self._send_json({"ошибка": str(e)}, code=500)
                return
        if self.path == "/api/save-settings":
            try:
                self._send_json(handle_save_settings(self._read_json()))
                return
            except Exception as e:
                self._send_json({"ошибка": str(e)}, code=500)
                return
        if self.path == "/api/apply-role":
            try:
                self._send_json(handle_apply_role(self._read_json()))
                return
            except Exception as e:
                self._send_json({"ошибка": str(e)}, code=500)
                return
        if self.path == "/api/ollama/ensure":
            try:
                self._send_json(ensure_ollama())
                return
            except Exception as e:
                self._send_json({"ошибка": str(e)}, code=500)
                return
        if self.path == "/api/check-key":
            try:
                self._send_json(check_api_key(self._read_json()))
                return
            except Exception as e:
                self._send_json({"ошибка": str(e)}, code=500)
                return
        self.send_error(404)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    info = provider_info()
    print("ИИ-ассистент запущен: http://%s:%d" % (HOST, PORT))
    print("Нейросеть: %s | модель: %s" % (info["провайдер"], info["модель"] or "—"))
    print("Настройки из: %s" % SETTINGS_CSV)
    print("Посты сохраняются в: %s" % OUTPUT_CSV)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
