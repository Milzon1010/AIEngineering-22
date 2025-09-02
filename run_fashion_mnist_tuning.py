import os, time, argparse, json
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.random.set_seed(7); np.random.seed(7)

def load_data(val_size=0.1, limit=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    if limit:
        x_train, y_train = x_train[:limit], y_train[:limit]
        x_test, y_test   = x_test[:max(limit//5, 1000)], y_test[:max(limit//5, 1000)]
    x_train = x_train.astype("float32")/255.0
    x_test  = x_test.astype("float32")/255.0
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=7, stratify=y_train)
    x_train = np.expand_dims(x_train, -1); x_val = np.expand_dims(x_val, -1); x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_model(hidden_units=128, dropout=0.2, extra_layer=False):
    inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    if extra_layer: x = tf.keras.layers.Dense(hidden_units//2, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def train_one(cfg, data, epochs=10, batch_size=32, patience=3):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    model = build_model(cfg.get("hidden_units",128), cfg.get("dropout",0.2), cfg.get("extra_layer",False))
    lr = cfg.get("lr",1e-3); opt_name = cfg.get("optimizer","adam").lower()
    if opt_name=="adam": opt = tf.keras.optimizers.Adam(lr)
    elif opt_name=="rmsprop": opt = tf.keras.optimizers.RMSprop(lr)
    elif opt_name=="sgd": opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
    else: raise ValueError(f"Unsupported optimizer: {opt_name}")
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)]
    t0=time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=2)
    train_time = time.time()-t0
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    best_epoch = int(np.argmax(hist.history["val_accuracy"]))
    return {
        "history": hist.history,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "best_val_acc": float(np.max(hist.history["val_accuracy"])),
        "best_train_acc": float(hist.history["accuracy"][best_epoch]),
        "overfit_gap": float(hist.history["accuracy"][best_epoch]-np.max(hist.history["val_accuracy"])),
        "epochs_trained": len(hist.history["loss"]),
        "train_time_sec": float(train_time),
    }, model

def plot_history(histories, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for name,h in histories.items(): plt.plot(h["history"]["val_accuracy"], label=name)
    plt.title("Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Val Acc"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"val_accuracy.png")); plt.close()
    plt.figure()
    for name,h in histories.items(): plt.plot(h["history"]["val_loss"], label=name)
    plt.title("Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("Val Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"val_loss.png")); plt.close()

def main():
    import argparse, json, pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    data = load_data(limit=args.limit)

    experiments = {
        "baseline":    {"lr":1e-3, "hidden_units":128, "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "lr_small":    {"lr":1e-4, "hidden_units":128, "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "lr_large":    {"lr":1e-2, "hidden_units":128, "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "hidden_64":   {"lr":1e-3, "hidden_units":64,  "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "hidden_256":  {"lr":1e-3, "hidden_units":256, "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "dropout_0p1": {"lr":1e-3, "hidden_units":128, "dropout":0.1, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "dropout_0p5": {"lr":1e-3, "hidden_units":128, "dropout":0.5, "optimizer":"adam", "extra_layer":False, "batch":args.batch},
        "batch_16":    {"lr":1e-3, "hidden_units":128, "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":16},
        "batch_128":   {"lr":1e-3, "hidden_units":128, "dropout":0.2, "optimizer":"adam", "extra_layer":False, "batch":128},
        "extra_layer": {"lr":1e-3, "hidden_units":128, "dropout":0.2, "optimizer":"adam", "extra_layer":True,  "batch":args.batch},
    }

    all_results, histories = [], {}
    for name, cfg in experiments.items():
        print(f"\n=== {name} ===")
        res,_ = train_one(cfg, data, epochs=args.epochs, batch_size=cfg.get("batch", args.batch))
        row = dict(name=name, **cfg, **{k:res[k] for k in ["test_acc","test_loss","best_val_acc","best_train_acc","overfit_gap","epochs_trained","train_time_sec"]})
        all_results.append(row); histories[name]=res

    os.makedirs("plots", exist_ok=True)
    df = pd.DataFrame(all_results).sort_values("test_acc", ascending=False)
    df.to_csv("results.csv", index=False)
    plot_history(histories, outdir="plots")

    best = df.iloc[0].to_dict()
    lines = [
        "# AIEngineering-22 — Hyperparameter Tuning Report",
        "","## Summary Table", df.to_markdown(index=False),
        "","## Best Configuration","```json", json.dumps(best, indent=2), "```",
        "","## Insights (fill me)",
        "- Most impactful hyperparameter: ...",
        "- Training dynamics: ...",
        f"- Final test accuracy: {best['test_acc']:.4f}",
        "","_Generated by run_fashion_mnist_tuning.py_"
    ]
    open("report.md","w").write("\n".join(lines))
    print("Saved results.csv, report.md, plots/.")

if __name__ == "__main__":
    main()
