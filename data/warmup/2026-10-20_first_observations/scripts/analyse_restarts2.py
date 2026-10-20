import json, collections, sys
sp = sys.argv[1]
ids = [1, 2, 3, 5, 6]
runs = {i: json.load(open(rf"{sp}\restart_{i}.json", encoding="utf-8")) for i in ids}
print("cache_n values seen:", {r["cache_n"] for rows in runs.values() for r in rows})
print()
print("request position -> which distinct reply (A/B/..) per restart, per prompt")
labels = {}
for label in ("amb", "plain"):
    texts = collections.Counter(r["text"] for rows in runs.values() for r in rows if r["label"] == label)
    names = {t: chr(65 + k) for k, (t, _) in enumerate(texts.most_common())}
    labels[label] = names
    print(f"{label}: {len(texts)} distinct replies; counts {[c for _, c in texts.most_common()]}")
print()
print("restart | req1 amb | req2 plain | req3 amb | req4 plain | req5 plain | req6 plain | req7 amb")
for i in ids:
    print(f"{i:>7} | " + " | ".join(f"{labels[r['label']][r['text']]:^8}" for r in runs[i]))
