"""Optional language interface for an already running local inference server."""

from http.client import HTTPConnection, HTTPException
import json
import re


INSTRUCTIONS = (
    "You are the language assistant in NNM Studio. The NNM-49 checkpoint is a scalar "
    "regressor. You handle conversation and drafting. Give concise, useful answers. "
    "For prose return one JSON object with action='reply' and a text string. "
    "For a prediction from the NNM use action='predict' and numeric x. "
    "For a request to create a runnable expression or evaluate it, return "
    "action='compile' or action='run', expression as a string, and x as a number "
    "if running. The compiler accepts one expression using numeric literals, x, "
    "parentheses, addition, subtraction, multiplication, and unary minus. "
    "Use ordinary double-quoted JSON with no surrounding Markdown. "
    "If you cannot express the request in that grammar, explain the limitation or "
    "draft the idea in prose. Do not claim you have executed or tested a draft."
)


class LocalLanguage:
    def __init__(self, port=8081, model="cognitive-qwen3-1.7b"):
        if not 1 <= port <= 65535:
            raise ValueError("The language model port must be between 1 and 65535.")
        self.port = port
        self.model = model

    def __call__(self, messages):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": INSTRUCTIONS}, *messages],
            "temperature": 0.2,
            "max_tokens": 768,
            "stream": False,
        }).encode("utf-8")
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request("POST", "/v1/chat/completions", body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f"The local language server returned HTTP {response.status}.")
            raw = response.read(131_073)
        except (OSError, HTTPException):
            raise ValueError(f"Start the local language server on port {self.port} to use general chat.") from None
        finally:
            connection.close()
        if len(raw) > 131_072:
            raise ValueError("The local language response was too large.")
        try:
            text = json.loads(raw)["choices"][0]["message"]["content"]
        except (ValueError, IndexError, KeyError, TypeError, RecursionError):
            raise ValueError("The local language server returned an unreadable response.") from None
        if not isinstance(text, str):
            raise ValueError("The local language server returned no text.")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not text or text.startswith("<think>"):
            raise ValueError("The local model did not finish its answer. Try a shorter prompt.")
        return re.sub(r"\A\x60{3}(?:json)?\s*(.*?)\s*\x60{3}\Z", r"\1", text,
                      flags=re.DOTALL | re.IGNORECASE)
