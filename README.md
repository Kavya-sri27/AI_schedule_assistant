# Agentic RAG Schedule Assistant

A chat-based agent that manages a user's schedule for the next 30 days, backed by a
RAG (retrieval-augmented generation) pipeline over a vector store, with two tools the
agent decides between on its own: `get_schedule` and `update_schedule`.

## Architecture

```
data/generate_schedule.py   -> creates 30 days of sample events (meetings, workshops,
                                tasks, appointments) -> data/schedule.json (source of truth)

vector_store.py             -> RAG vector index over the schedule.
                                - Primary backend: ChromaDB (persistent, local)
                                - Fallback backend: pure-numpy TF-IDF cosine index,
                                  used automatically if chromadb isn't installed, so
                                  the app runs anywhere with zero external services.

tools.py                    -> the two agent tools
                                - get_schedule(query, date, start_time, end_time)
                                - update_schedule(action, ...) for add / update / remove
                                Both read/write the JSON store AND the vector index,
                                so retrieval always reflects the latest state.

agent.py                    -> the agentic decision layer
                                - LLM mode: if ANTHROPIC_API_KEY is set, uses Claude's
                                  native tool-use (Messages API) so the model itself
                                  decides which tool to call and with what arguments —
                                  this is the "real" agent.
                                - Fallback mode: deterministic intent parser covering
                                  the example queries, used when no API key is set.

app.py + templates/index.html -> Flask chat UI + /api/chat endpoint
```

## Run locally

```bash
pip install -r requirements.txt
python data/generate_schedule.py     # regenerate sample data any time
export ANTHROPIC_API_KEY=sk-ant-...  # optional — enables true LLM tool-calling
python app.py
# open http://localhost:5000
```

Without `ANTHROPIC_API_KEY` set, the app still runs completely, using the rule-based
fallback parser — useful for grading/demo environments with no API key or outbound
network access.

## Example queries

- "What do I have scheduled tomorrow?"
- "Am I free Friday afternoon?"
- "Add a meeting with the design team on August 20 at 3 PM"
- "Move my meeting from 2 PM to 4 PM"
- "Cancel my dentist appointment"

## Deploying to get a public URL

This project wasn't deployed from within this sandboxed build environment (it has no
outbound internet access, so it can't reach a hosting provider). Any of these get you
a live URL in a few minutes:

**Render (easiest)**
1. Push this folder to a GitHub repo.
2. render.com → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add environment variable `ANTHROPIC_API_KEY` (optional but recommended).
6. Deploy — Render gives you a `https://<name>.onrender.com` URL.

**Railway**
1. railway.app → New Project → Deploy from GitHub repo.
2. Railway auto-detects the `Procfile`. Add `ANTHROPIC_API_KEY` under Variables.
3. Deploy — copy the generated public domain.

**Hugging Face Spaces (Docker/Flask template)** also works well and is free.

Once deployed, put the URL in `deployment_url.txt` (included in this bundle).

## Notes on the vector DB choice

ChromaDB is used as the primary backend because it needs no external API key/service —
it runs embedded in the app process (like SQLite for vectors), which fits a self-contained
30-day personal schedule well. Swapping to Pinecone only requires implementing the same
`.upsert()/.query()/.delete()` interface in `vector_store.py` with the Pinecone client —
no changes needed anywhere else.
