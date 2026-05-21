import urllib.request, json
with open("cicids_test_vectors.txt") as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
for l in lines:
    if "Web" in l:
        parts = l.split("|")
        feats = [float(x) for x in parts[1].split(",")]
        data = json.dumps({"features": feats}).encode()
        req = urllib.request.Request("http://127.0.0.1:8080/predict", data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(f"True: {repr(parts[0])}")
        print(f"Pred: {repr(result['label'])}")
        print(f"Match: {result['label'] == parts[0]}")
        print(f"Pred hex: {result['label'].encode('utf-8').hex()}")
        print()
        break
