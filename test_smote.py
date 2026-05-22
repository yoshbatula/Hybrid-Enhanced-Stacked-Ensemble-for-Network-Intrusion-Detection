import urllib.request, json

with open("smote_test_vectors.txt") as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

print("Testing SMOTE-synthetic CICIDS samples...\n")
correct = 0
total = 0
for line in lines:
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
    print(f"{label:35s} -> {result['label']:30s} (conf={result['confidence']:.4f}) [{status}]")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
