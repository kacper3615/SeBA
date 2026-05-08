import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ----- Single models -----


def validate_classifier(P, clf, val_loader):
    clf.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total_cnt = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(P.device), y.to(P.device)
            logits = clf(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_cnt += X.size(0)

    clf.train()

    avg_loss = total_loss / total_cnt if total_cnt > 0 else 0.0
    avg_acc = total_correct / total_cnt if total_cnt > 0 else 0.0
    return avg_loss, avg_acc


def validate_proto(P, f_model, x_sup, y_sup, qry_loader, metric="euclidean"):
    criterion = nn.CrossEntropyLoss()
    f_model.eval()

    with torch.no_grad():
        z_sup = f_model(x_sup)
        classes, _ = torch.sort(torch.unique(y_sup))
        protos = torch.stack([z_sup[y_sup == c].mean(0) for c in classes], dim=0)

        label_to_idx = {int(c.item()): i for i, c in enumerate(classes)}

        total_loss = 0.0
        total_correct = 0
        total_cnt = 0

        for X, y in qry_loader:
            X = X.to(P.device)
            y = y.to(P.device)

            z_q = f_model(X)

            if metric == "euclidean":
                dists = torch.cdist(z_q, protos, p=2)
                logits = -dists
            elif metric == "cosine":
                sims = F.cosine_similarity(
                    z_q.unsqueeze(1), protos.unsqueeze(0), dim=-1
                )
                logits = sims

            y_ce = torch.tensor(
                [label_to_idx[int(t.item())] for t in y], device=P.device
            )

            loss = criterion(logits, y_ce)
            preds = logits.argmax(dim=1)
            correct = (preds == y_ce).sum().item()

            total_loss += loss.item() * X.size(0)
            total_correct += correct
            total_cnt += X.size(0)

    avg_loss = total_loss / total_cnt if total_cnt > 0 else 0.0
    avg_acc = total_correct / total_cnt if total_cnt > 0 else 0.0
    return avg_loss, avg_acc


def validate_nn(
    P,
    f_model,
    x_sup: torch.Tensor,
    y_sup: torch.Tensor,
    qry_loader,
    metric: str = "euclidean",
):

    nll = nn.NLLLoss()

    f_model.eval()
    x_sup = x_sup.to(P.device)
    y_sup = y_sup.to(P.device)

    with torch.no_grad():
        z_sup = f_model(x_sup).detach()

    knn = KNeighborsClassifier(
        n_neighbors=1, metric=metric, weights="uniform", algorithm="auto"
    )
    knn.fit(z_sup.cpu().numpy(), y_sup.cpu().numpy())

    classes_np = knn.classes_
    label_to_idx = {int(c): i for i, c in enumerate(classes_np)}

    total_loss = total_correct = total_cnt = 0

    with torch.no_grad():
        for X, y in qry_loader:
            X = X.to(P.device)
            y = y.to(P.device)

            z_q = f_model(X).detach()
            probs_np = knn.predict_proba(z_q.cpu().numpy())
            probs = torch.from_numpy(probs_np).to(P.device)

            y_idx = torch.tensor(
                [label_to_idx[int(t.item())] for t in y], device=P.device
            )
            loss = nll((probs.clamp_min(1e-12)).log(), y_idx)

            preds_idx = probs.argmax(dim=1)
            total_loss += loss.item() * X.size(0)
            total_correct += (preds_idx == y_idx).sum().item()
            total_cnt += X.size(0)

    f_model.train()

    avg_loss = total_loss / total_cnt if total_cnt > 0 else 0.0
    avg_acc = total_correct / total_cnt if total_cnt > 0 else 0.0
    return avg_loss, avg_acc


# ----- Ensemble models -----


def validate_classifier_ensemble(P, clfs, val_loader):
    for clf in clfs:
        clf.eval()

    ce = nn.CrossEntropyLoss()
    total_loss = total_correct = total_cnt = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(P.device), y.to(P.device)

            logits = torch.stack([clf(X) for clf in clfs], dim=0).mean(dim=0)
            loss = ce(logits, y)
            preds = logits.argmax(1)

            # probs = torch.stack([F.softmax(clf(X), dim=1) for clf in clfs], dim=0).mean(dim=0)
            # loss = nll(torch.log(probs.clamp_min(1e-12)), y)
            # preds = probs.argmax(1)

            total_loss += loss.item() * X.size(0)
            total_correct += (preds == y).sum().item()
            total_cnt += X.size(0)

    for clf in clfs:
        clf.train()

    avg_loss = total_loss / total_cnt if total_cnt > 0 else 0.0
    avg_acc = total_correct / total_cnt if total_cnt > 0 else 0.0
    return avg_loss, avg_acc


def validate_proto_ensemble(P, f_models, x_sup, y_sup, qry_loader, metric="euclidean"):
    ce = nn.CrossEntropyLoss()

    for fm in f_models:
        fm.eval()

    x_sup = x_sup.to(P.device)
    y_sup = y_sup.to(P.device)

    with torch.no_grad():
        classes, _ = torch.sort(torch.unique(y_sup))
        label_to_idx = {int(c.item()): i for i, c in enumerate(classes)}

        protos_per_model = []
        for fm in f_models:
            z_sup = fm(x_sup)
            protos = torch.stack([z_sup[y_sup == c].mean(0) for c in classes], dim=0)
            protos_per_model.append(protos)

        total_loss = 0.0
        total_correct = 0
        total_cnt = 0

        for X, y in qry_loader:
            X = X.to(P.device)
            y = y.to(P.device)

            logits_list = []
            for fm, protos in zip(f_models, protos_per_model):
                z_q = fm(X)
                if metric == "euclidean":
                    dists = torch.cdist(z_q, protos, p=2)
                    logits = -dists
                elif metric == "cosine":
                    sims = F.cosine_similarity(
                        z_q.unsqueeze(1), protos.unsqueeze(0), dim=-1
                    )
                    logits = sims
                logits_list.append(logits)

            logits_ens = torch.stack(logits_list, dim=0).mean(dim=0)

            y_ce = torch.tensor(
                [label_to_idx[int(t.item())] for t in y], device=P.device
            )

            loss = ce(logits_ens, y_ce)
            preds = logits_ens.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            total_correct += (preds == y_ce).sum().item()
            total_cnt += X.size(0)

    for fm in f_models:
        fm.train()

    avg_loss = total_loss / total_cnt if total_cnt > 0 else 0.0
    avg_acc = total_correct / total_cnt if total_cnt > 0 else 0.0
    return avg_loss, avg_acc


def validate_nn_ensemble(
    P,
    f_models,
    x_sup: torch.Tensor,
    y_sup: torch.Tensor,
    qry_loader,
    metric: str = "euclidean",
):

    nll = nn.NLLLoss()

    for fm in f_models:
        fm.eval()

    x_sup = x_sup.to(P.device)
    y_sup = y_sup.to(P.device)

    with torch.no_grad():
        classes_all = torch.sort(torch.unique(y_sup)).values.cpu().numpy()

        knns = []
        for fm in f_models:
            z_sup = fm(x_sup).detach().cpu().numpy()
            knn = KNeighborsClassifier(
                n_neighbors=1, metric=metric, weights="uniform", algorithm="auto"
            )
            knn.fit(z_sup, y_sup.cpu().numpy())
            knns.append(knn)

    total_loss = total_correct = total_cnt = 0

    with torch.no_grad():
        for X, y in qry_loader:
            X = X.to(P.device)
            y = y.to(P.device)

            probs_accum = None

            for fm, knn in zip(f_models, knns):
                z_q = fm(X).detach().cpu().numpy()
                probs_m_np = knn.predict_proba(z_q)

                colmap = {c: i for i, c in enumerate(knn.classes_)}
                B = probs_m_np.shape[0]
                C_all = len(classes_all)
                aligned = np.zeros((B, C_all), dtype=probs_m_np.dtype)
                for j, c in enumerate(classes_all):
                    if c in colmap:
                        aligned[:, j] = probs_m_np[:, colmap[c]]

                probs_m = torch.from_numpy(aligned).to(P.device)

                if probs_accum is None:
                    probs_accum = probs_m
                else:
                    probs_accum = probs_accum + probs_m

            probs = probs_accum / len(f_models)
            probs = probs.clamp_min(1e-12)

            idx_map = {int(c): i for i, c in enumerate(classes_all)}
            y_idx = torch.tensor([idx_map[int(t.item())] for t in y], device=P.device)

            loss = nll(probs.log(), y_idx)
            preds_idx = probs.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            total_correct += (preds_idx == y_idx).sum().item()
            total_cnt += X.size(0)

    for fm in f_models:
        fm.train()

    avg_loss = total_loss / total_cnt if total_cnt > 0 else 0.0
    avg_acc = total_correct / total_cnt if total_cnt > 0 else 0.0
    return avg_loss, avg_acc
