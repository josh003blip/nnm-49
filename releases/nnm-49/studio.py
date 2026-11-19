"""Run the standalone NNM-49 chat window or ask a question from the terminal."""

import argparse
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread

from local_language import LocalLanguage
from research_chat import CHECKPOINT, ResearchChat


EXAMPLES = (
    ("Predict", "What would the model predict for 0.5?"),
    ("Compile & run", "Run x*x + 2 at 3"),
    ("Fit an equation", "Fit an equation to [(0,1), (1,3), (2,5), (3,7)]; test on [(4,9), (5,11)]"),
    ("Time question", "Maya was painting from 2:00 to 2:30. At 2:12, was she painting?"),
)


def gui(chat):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("NNM Studio")
    root.geometry("790x630")
    root.minsize(490, 400)
    root.configure(bg="#f6f5f9")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(2, weight=1)

    ink, accent, secondary, canvas = "#232339", "#4b3f9a", "#5d6073", "#f6f5f9"
    header = tk.Frame(root, bg="#eeebf6", padx=20, pady=13)
    header.grid(row=0, column=0, sticky="ew")
    tk.Label(header, text="NNM Studio", font=("Segoe UI", 20, "bold"),
             fg=ink, bg="#eeebf6").pack(side="left")
    new = ttk.Button(header, text="New chat")
    new.pack(side="right")

    bar = tk.Frame(root, bg=canvas, padx=20, pady=10)
    bar.grid(row=1, column=0, sticky="ew")
    shortcuts = []

    pane = tk.Frame(root, bg=canvas, padx=20)
    pane.grid(row=2, column=0, sticky="nsew")
    pane.columnconfigure(0, weight=1)
    pane.rowconfigure(0, weight=1)
    transcript = tk.Text(pane, wrap="word", padx=15, pady=10, relief="flat",
                         font=("Segoe UI", 11), bg="white", fg=ink, state="disabled")
    transcript.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(pane, orient="vertical", command=transcript.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    transcript.configure(yscrollcommand=scrollbar.set)
    transcript.tag_configure("label", foreground=accent, font=("Segoe UI", 11, "bold"))

    composer = tk.Frame(root, bg=canvas, padx=20, pady=9)
    composer.grid(row=3, column=0, sticky="ew")
    composer.columnconfigure(0, weight=1)
    prompt = tk.Text(composer, height=3, wrap="word", padx=9, pady=5, relief="flat",
                     font=("Segoe UI", 11), bg="white", fg=ink)
    prompt.grid(row=0, column=0, sticky="ew")
    send = ttk.Button(composer, text="Send")
    send.grid(row=0, column=1, padx=(8, 0), sticky="s")
    mode = "Local language model enabled" if chat.language else "Offline research mode"
    status = tk.StringVar(value=f"{mode}  ·  Enter to send")
    tk.Label(root, textvariable=status, bg=canvas, fg=secondary, anchor="w",
             padx=20, pady=5).grid(row=4, column=0, sticky="ew")

    replies = Queue()

    def show(label, text):
        transcript.configure(state="normal")
        transcript.insert("end", label + "\n", "label")
        transcript.insert("end", text + "\n\n")
        transcript.configure(state="disabled")
        transcript.see("end")

    def reset():
        chat.reset()
        transcript.configure(state="normal")
        transcript.delete("1.0", "end")
        transcript.configure(state="disabled")
        prompt.delete("1.0", "end")
        show("NNM Studio", "Ask for a prediction, compile a formula, or fit one to your data. "
             "Choose an example to get started.")
        prompt.focus_set()

    def fill(text):
        prompt.delete("1.0", "end")
        prompt.insert("1.0", text)
        prompt.focus_set()

    def busy(enabled):
        for button in (send, new, *shortcuts):
            button.configure(state="disabled" if enabled else "normal")
        prompt.configure(state="disabled" if enabled else "normal")
        status.set("Working…" if enabled else f"{mode}  ·  Enter to send")

    def work(text):
        try:
            replies.put(chat.reply(text))
        except Exception as error:
            replies.put(f"Couldn’t complete that request. {error}")

    def poll():
        try:
            answer = replies.get_nowait()
        except Empty:
            root.after(40, poll)
            return
        show("NNM Studio", answer)
        busy(False)
        prompt.focus_set()

    def submit(event=None):
        text = prompt.get("1.0", "end-1c").strip()
        if text and str(send.cget("state")) != "disabled":
            prompt.delete("1.0", "end")
            show("You", text)
            busy(True)
            Thread(target=work, args=(text,), daemon=True).start()
            root.after(40, poll)
        return "break"

    for label, text in EXAMPLES:
        button = ttk.Button(bar, text=label, command=lambda value=text: fill(value))
        button.pack(side="left", padx=(0, 7))
        shortcuts.append(button)
    send.configure(command=submit)
    new.configure(command=reset)
    prompt.bind("<Return>", submit)
    prompt.bind("<Shift-Return>", lambda event: None)
    reset()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--local-llm", action="store_true", help="Enable chat through a local model server.")
    parser.add_argument("--llm-port", type=int, default=8081)
    parser.add_argument("--llm-model", default="cognitive-qwen3-1.7b")
    parser.add_argument("--ask", action="append", help="Ask in the terminal. Repeat for follow-up turns.")
    args = parser.parse_args()
    try:
        language = LocalLanguage(args.llm_port, args.llm_model) if args.local_llm else None
        chat = ResearchChat(checkpoint=args.checkpoint, language=language)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.ask:
        sys.stdout.reconfigure(errors="backslashreplace")
        for text in args.ask:
            print(chat.reply(text))
    else:
        gui(chat)


if __name__ == "__main__":
    main()
