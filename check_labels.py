import joblib
le = joblib.load("saved_model/label_encoder.pkl")
for i, name in enumerate(le.classes_):
    print(f"{i:2d}: {name.encode('utf-8').hex()} | {repr(name)}")
