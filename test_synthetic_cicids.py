import urllib.request, json

for fname, desc in [("synthetic_test_vectors.txt", "HAND-CRAFTED"), ("cicids_test_vectors.txt", "CICIDS 2017")]:
    print(f"\n=== {desc} ({fname}) ===")
    with open(fname) as f:
        lines = f.readlines()
    correct = 0
    total = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        label = parts[0]
        feats = [float(x) for x in parts[1].split(",")]
        data = json.dumps({"features": feats}).encode()
        req = urllib.request.Request("http://127.0.0.1:8080/predict", data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        match = result["label"].lower().replace(" ", "") == label.lower().replace(" ", "")
        if match:
            correct += 1
        total += 1
        status = "OK" if match else "MISMATCH"
        print(f"  {label:35s} -> {result['label']:30s} (conf={result['confidence']:.4f}) [{status}]")
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
