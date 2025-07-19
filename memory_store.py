import json, os, time

LT_FILE = "memory/long_term.jsonl"

class MemoryStore:
    def __init__(self, path=LT_FILE):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # We WON'T load entire file into RAM if it grows huge; small project so fine.
        self._cache = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        self._cache.append(json.loads(line))
                    except:
                        pass

    def add_fact(self, fact:str):
        fact = fact.strip()
        if not fact: return
        entry = {"ts": time.time(), "fact": fact}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False)+"\n")
        self._cache.append(entry)

    def list_facts(self, limit=50):
        return self._cache[-limit:]

    def delete_index(self, idx:int):
        # idx is display index from list_facts order (most recent last)
        facts = self.list_facts()
        if idx < 0 or idx >= len(facts):
            return False
        target = facts[idx]
        # rebuild file
        self._cache = [e for e in self._cache if e is not target]
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self._cache:
                f.write(json.dumps(e, ensure_ascii=False)+"\n")
        return True

    def retrieve(self, user_text:str, max_items:int=2):
        """
        Very light retrieval: word overlap.
        """
        words = {w.lower() for w in user_text.split() if len(w) > 3}
        scored = []
        for e in self._cache:
            fact_words = set(e["fact"].lower().split())
            overlap = words & fact_words
            if overlap:
                scored.append((len(overlap), e["fact"]))
        scored.sort(reverse=True)
        top = [fact for _, fact in scored[:max_items]]
        if not top:
            # fallback: recent facts
            recent = [e["fact"] for e in self._cache[-max_items:]]
            top = recent
        # Deduplicate order-preserving
        out = []
        for f in top:
            if f not in out:
                out.append(f)
        return out[:max_items]
