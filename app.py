import os

from flask import Flask, jsonify, render_template, request

from agent import Agent

app = Flask(__name__)
agent = Agent()

# Very small in-memory per-process conversation history (fine for a demo / single user).
_history = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    global _history
    user_message = (request.json or {}).get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a message."})
    reply = agent.handle_message(user_message, history=_history)
    _history.append({"role": "user", "content": user_message})
    _history.append({"role": "assistant", "content": reply})
    _history = _history[-20:]
    return jsonify({"reply": reply})


@app.route("/api/reset", methods=["POST"])
def reset():
    global _history
    _history = []
    return jsonify({"status": "ok"})


@app.route("/healthz")
def healthz():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
