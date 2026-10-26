"""Candidate detectors of config/models.yaml (amendment 1.4): one interface for every family.

    m = make(name, spec, params, seed)
    m.fit(X_train, y_train, X_val, y_val)     # numpy float32, y in {0, 1}; val only for early stopping
    p = m.predict_proba(X)                    # P(attack)
    sha = m.save(path)

Input preparation (`Prep`) is fitted on the training rows only and saved inside the model file:
median imputation for every family, standardisation for the linear model, sign(x) * log1p(|x|) then
standardisation for neural models (heavy-tailed flow counters otherwise dominate the gradients).
"""
import hashlib
import io
import pickle
import time

import numpy as np

NEURAL = ("mlp", "cnn_lstm")


class Prep:
    def __init__(self, family):
        self.family = family

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        Z = self._base(X)
        if self.family in ("linear", "neural"):
            self.mean, self.std = Z.mean(axis=0), Z.std(axis=0)
            self.std = np.where(self.std > 0, self.std, 1.0)
        return self

    def _base(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, self.median)
        if self.family == "neural":
            X = np.sign(X) * np.log1p(np.abs(X))
        return X

    def transform(self, X):
        Z = self._base(X)
        if self.family in ("linear", "neural"):
            Z = (Z - self.mean) / self.std
        return Z.astype(np.float32)


def _class_weight_ratio(y):
    pos = max(int((y == 1).sum()), 1)
    return float((y == 0).sum()) / pos


class Candidate:
    def __init__(self, name, spec, params, seed):
        self.name, self.spec, self.params, self.seed = name, spec, dict(params), seed
        self.family = spec["family"]
        self.prep = Prep(self.family)
        self.info = {}

    # ---- fitting --------------------------------------------------------------------------------------------
    def fit(self, X, y, Xv, yv):
        t0 = time.time()
        Xp = self.prep.fit(X).transform(X)
        Xvp = self.prep.transform(Xv)
        y, yv = np.asarray(y, dtype=np.int64), np.asarray(yv, dtype=np.int64)
        getattr(self, f"_fit_{self.name}")(Xp, y, Xvp, yv)
        self.info["train_seconds"] = round(time.time() - t0, 1)
        return self

    def _fit_logreg(self, X, y, Xv, yv):
        from sklearn.linear_model import LogisticRegression
        f = self.spec.get("fixed", {})
        self.model = LogisticRegression(C=self.params["C"], class_weight=self.params["class_weight"],
                                        max_iter=f.get("max_iter", 2000), solver=f.get("solver", "lbfgs"),
                                        random_state=self.seed).fit(X, y)
        self.info["iterations"] = int(np.max(self.model.n_iter_))

    def _fit_random_forest(self, X, y, Xv, yv):
        from sklearn.ensemble import RandomForestClassifier
        f = self.spec.get("fixed", {})
        self.model = RandomForestClassifier(n_estimators=f.get("n_estimators", 300),
                                            max_features=f.get("max_features", "sqrt"), n_jobs=f.get("n_jobs", -1),
                                            max_depth=self.params["max_depth"],
                                            min_samples_leaf=self.params["min_samples_leaf"],
                                            class_weight=self.params["class_weight"],
                                            random_state=self.seed).fit(X, y)

    def _fit_xgboost(self, X, y, Xv, yv):
        from xgboost import XGBClassifier
        f = dict(self.spec.get("fixed", {}))
        spw = self.params["scale_pos_weight"]
        spw = _class_weight_ratio(y) if spw == "balanced" else float(spw)
        self.model = XGBClassifier(**f, max_depth=self.params["max_depth"],
                                   learning_rate=self.params["learning_rate"], scale_pos_weight=spw,
                                   random_state=self.seed)
        self.model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.info["best_iteration"] = int(self.model.best_iteration)

    def _fit_mlp(self, X, y, Xv, yv):
        self._fit_torch(X, y, Xv, yv)

    def _fit_cnn_lstm(self, X, y, Xv, yv):
        self._fit_torch(X, y, Xv, yv)

    # ---- neural models --------------------------------------------------------------------------------------
    def _net(self, n_features):
        import torch.nn as nn
        f, p = self.spec.get("fixed", {}), self.params
        if self.name == "mlp":
            layers, width, d_in = [], p["width"], n_features
            for i in range(f.get("hidden", 3)):
                w = max(width // (2 ** i), 16)
                layers += [nn.Linear(d_in, w), nn.ReLU(), nn.Dropout(p["dropout"])]
                d_in = w
            return nn.Sequential(*layers, nn.Linear(d_in, 1))
        return CNNLSTM(n_features, f.get("conv_channels", [32, 64]), f.get("kernel", 3), f.get("lstm_units", 64),
                       p["dropout"])

    def _fit_torch(self, X, y, Xv, yv):
        import torch
        from ..utils.seed import set_seed
        set_seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f = self.spec.get("fixed", {})
        net = self._net(X.shape[1]).to(dev)
        pw = _class_weight_ratio(y) if self.params.get("class_weight") == "balanced" else 1.0
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))
        opt = torch.optim.Adam(net.parameters(), lr=f.get("lr", 1e-3), weight_decay=f.get("weight_decay", 1e-4))
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        gen = torch.Generator().manual_seed(self.seed)
        bs, best, best_state, bad, hist = f.get("batch_size", 4096), np.inf, None, 0, []
        for epoch in range(f.get("epochs", 50)):
            net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx].to(dev)).squeeze(1), yt[idx].to(dev))
                loss.backward()
                opt.step()
            vl = self._val_loss(net, Xv, yv, loss_fn, dev)
            hist.append(round(vl, 6))
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= f.get("patience", 5):
                    break
        net.load_state_dict(best_state)
        self.model = net.cpu().eval()
        self.info.update({"epochs_run": len(hist), "best_epoch": int(np.argmin(hist)) + 1, "val_loss": hist,
                          "device": str(dev)})

    @staticmethod
    def _val_loss(net, Xv, yv, loss_fn, dev, bs=65536):
        import torch
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), bs):
                xb = torch.from_numpy(Xv[i:i + bs]).to(dev)
                yb = torch.from_numpy(yv[i:i + bs].astype(np.float32)).to(dev)
                tot += float(loss_fn(net(xb).squeeze(1), yb)) * len(xb)
        return tot / len(Xv)

    # ---- prediction and storage -----------------------------------------------------------------------------
    def predict_proba(self, X, bs=262144):
        Xp = self.prep.transform(X)
        if self.name in NEURAL:
            import torch
            out = []
            with torch.no_grad():
                for i in range(0, len(Xp), bs):
                    out.append(torch.sigmoid(self.model(torch.from_numpy(Xp[i:i + bs])).squeeze(1)).numpy())
            return np.concatenate(out).astype(np.float64) if out else np.zeros(0)
        return self.model.predict_proba(Xp)[:, 1].astype(np.float64)

    def save(self, path):
        """Pickle of the whole candidate (neural weights as a torch state dict); returns its SHA-256."""
        state = {"name": self.name, "spec": self.spec, "params": self.params, "seed": self.seed,
                 "prep": self.prep, "info": self.info}
        if self.name in NEURAL:
            import torch
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            state["torch_state"] = buf.getvalue()
            state["n_features"] = len(self.prep.median)
        else:
            state["model"] = self.model
        blob = pickle.dumps(state, protocol=4)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def load(cls, path):
        state = pickle.loads(path.read_bytes())
        m = cls(state["name"], state["spec"], state["params"], state["seed"])
        m.prep, m.info = state["prep"], state["info"]
        if m.name in NEURAL:
            import torch
            net = m._net(state["n_features"])
            net.load_state_dict(torch.load(io.BytesIO(state["torch_state"]), weights_only=True))
            m.model = net.eval()
        else:
            m.model = state["model"]
        return m


def _cnn_lstm_class():
    import torch.nn as nn

    class _CNNLSTM(nn.Module):
        """Features as a length-F sequence of one channel (a common IDS deep baseline)."""

        def __init__(self, n_features, channels, kernel, units, dropout):
            super().__init__()
            c1, c2 = channels
            self.conv = nn.Sequential(nn.Conv1d(1, c1, kernel, padding=kernel // 2), nn.ReLU(),
                                      nn.Conv1d(c1, c2, kernel, padding=kernel // 2), nn.ReLU(),
                                      nn.MaxPool1d(2) if n_features >= 4 else nn.Identity())
            self.lstm = nn.LSTM(c2, units, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(units, 1))

        def forward(self, x):
            h = self.conv(x.unsqueeze(1)).transpose(1, 2)
            _, (hn, _) = self.lstm(h)
            return self.head(hn[-1])

    return _CNNLSTM


def CNNLSTM(*args):
    return _cnn_lstm_class()(*args)


def make(name, spec, params, seed):
    if name not in ("logreg", "random_forest", "xgboost", "mlp", "cnn_lstm"):
        raise ValueError(f"unknown candidate {name}")
    return Candidate(name, spec, params, seed)


def best_threshold(y, p):
    """Threshold t maximising binary macro-F1 when predicting attack for p >= t (preregistration §5).

    Exact: every distinct score is a candidate cut, plus "nothing is an attack". Ties go to the higher threshold.
    """
    y, p = np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.float64)
    P, N = int(y.sum()), int(len(y) - y.sum())
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]       # end of each run of equal scores
    tp, fp, thr = tp[last], fp[last], ps[last]
    tp, fp, thr = np.r_[0, tp], np.r_[0, fp], np.r_[np.inf, thr]
    fn, tn = P - tp, N - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1a = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1b = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    mf = (f1a + f1b) / 2
    i = int(np.argmax(mf))
    return float(thr[i]), float(mf[i])
