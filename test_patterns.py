import urllib.request, json

def test(name, feats):
    data = json.dumps({"features": feats}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/predict", data=data, headers={"Content-Type":"application/json"})
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(f"{name:25s} -> {result['label']:25s} (conf={result['confidence']})")
    except urllib.error.HTTPError as e:
        print(f"{name:25s} -> ERROR: {e.read().decode()[:80]}")

tests = [
    ("All 1..78", list(range(1, 79))),
    ("All zeros", [0]*78),
    ("All 99999", [99999]*78),
    ("All 0.1", [0.1]*78),
    ("Dest port=80 only", [80] + [0]*77),
    ("High duration only", [0, 99999999] + [0]*76),
    ("High SYN flag", [0]*44 + [9999] + [0]*33),
    ("Many fwd packets", [80, 1000, 9999, 0] + [0]*74),
    ("Many fwd+bwd pkts", [80, 1000, 9999, 9999] + [0]*74),
    ("High bytes", [0]*4 + [9999999, 9999999] + [0]*72),
    ("Mix high values", [80, 9999999, 500, 400, 3000, 2000, 64, 0, 6, 2, 64, 0, 6, 2, 500, 100, 100000, 50000, 9999999, 100, 9999999, 50000, 25000, 9999999, 50, 500000, 25000, 10000, 500000, 1, 0, 0, 0, 0, 4000, 3000, 1, 1, 0, 64, 6, 12, 150, 0, 1, 0, 1, 0, 0, 0, 1, 500, 6, 6, 4000, 0, 0, 0, 0, 0, 0, 500, 3000, 400, 2000, 8192, 256, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0]),
]

for name, feats in tests:
    test(name, feats)
