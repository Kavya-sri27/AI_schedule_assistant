"""
Vector store layer for the RAG pipeline.

Primary backend: ChromaDB (persistent, local, no external API needed for embeddings —
uses Chroma's built-in sentence-transformer embedding function).

Fallback backend: a small pure-numpy TF-IDF cosine-similarity index. This is used
automatically if chromadb isn't installed in the current environment, so the whole
project still runs end-to-end for local development/demo purposes. Swap in ChromaDB
or Pinecone in production by installing the package — no application code changes
needed since both backends implement the same .upsert() / .query() / .delete() interface.
"""
import json
import math
import os
import re
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SCHEDULE_JSON = os.path.join(DATA_DIR, "schedule.json")
VECTORDB_DIR = os.path.join(os.path.dirname(__file__), "vectordb")


def event_to_text(e):
    return (
        f"{e['type'].capitalize()}: {e['title']} on {e['day_of_week']}, {e['date']} "
        f"from {e['start_time']} to {e['end_time']}. Notes: {e['notes']}"
    )


# ---------------------------------------------------------------------------
# Backend 1: ChromaDB (used automatically when the package is available)
# ---------------------------------------------------------------------------
class ChromaBackend:
    def __init__(self, collection_name="schedule"):
        import chromadb
        self.client = chromadb.PersistentClient(path=VECTORDB_DIR)
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert(self, event):
        self.collection.upsert(
            ids=[event["id"]],
            documents=[event_to_text(event)],
            metadatas=[event],
        )

    def delete(self, event_id):
        try:
            self.collection.delete(ids=[event_id])
        except Exception:
            pass

    def query(self, query_text, n_results=8, where=None):
        res = self.collection.query(
            query_texts=[query_text], n_results=n_results, where=where or {}
        )
        metadatas = res.get("metadatas", [[]])[0]
        return metadatas

    def all_events(self):
        res = self.collection.get()
        return res.get("metadatas", [])


# ---------------------------------------------------------------------------
# Backend 2: Local numpy/TF-IDF fallback (no external dependencies)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class LocalVectorBackend:
    """Minimal in-process vector index: TF-IDF vectors + cosine similarity.
    Persists to a JSON file so data survives restarts, mirroring a real vector DB."""

    def __init__(self, path=None):
        self.path = path or os.path.join(VECTORDB_DIR, "local_index.json")
        self._store = {}  # id -> {"metadata": event_dict, "text": str}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._store = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._store, f, indent=2)

    def upsert(self, event):
        self._store[event["id"]] = {"metadata": event, "text": event_to_text(event)}
        self._save()

    def delete(self, event_id):
        self._store.pop(event_id, None)
        self._save()

    def all_events(self):
        return [v["metadata"] for v in self._store.values()]

    def _tfidf_vectors(self, texts):
        docs_tokens = [_tokenize(t) for t in texts]
        df = Counter()
        for tokens in docs_tokens:
            for term in set(tokens):
                df[term] += 1
        n_docs = max(len(texts), 1)
        vectors = []
        for tokens in docs_tokens:
            tf = Counter(tokens)
            vec = {}
            for term, count in tf.items():
                idf = math.log((n_docs + 1) / (df[term] + 1)) + 1
                vec[term] = count * idf
            vectors.append(vec)
        return vectors

    @staticmethod
    def _cosine(v1, v2):
        common = set(v1) & set(v2)
        dot = sum(v1[t] * v2[t] for t in common)
        norm1 = math.sqrt(sum(x * x for x in v1.values())) or 1e-9
        norm2 = math.sqrt(sum(x * x for x in v2.values())) or 1e-9
        return dot / (norm1 * norm2)

    def query(self, query_text, n_results=8, where=None):
        items = list(self._store.values())
        if where:
            items = [it for it in items if all(it["metadata"].get(k) == v for k, v in where.items())]
        if not items:
            return []
        texts = [it["text"] for it in items] + [query_text]
        vectors = self._tfidf_vectors(texts)
        query_vec = vectors[-1]
        doc_vecs = vectors[:-1]
        scored = [
            (self._cosine(query_vec, dv), items[i]["metadata"])
            for i, dv in enumerate(doc_vecs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:n_results]]


# ---------------------------------------------------------------------------
# Public factory — chooses the best available backend automatically
# ---------------------------------------------------------------------------
def get_vector_store():
    try:
        import chromadb  # noqa: F401
        return ChromaBackend()
    except ImportError:
        return LocalVectorBackend()


def build_index_from_schedule(store, schedule_path=SCHEDULE_JSON):
    with open(schedule_path) as f:
        events = json.load(f)
    for e in events:
        store.upsert(e)
    return len(events)


if __name__ == "__main__":
    store = get_vector_store()
    count = build_index_from_schedule(store)
    print(f"Indexed {count} events into vector store ({type(store).__name__})")
