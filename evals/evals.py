from torch.utils.data import Subset, DataLoader

from models import Classifier
from trainers.trainer import train_classifier
from evals.validations import (
    validate_proto,
    validate_classifier,
    validate_nn,
    validate_classifier_ensemble,
    validate_proto_ensemble,
    validate_nn_ensemble,
)
from common.utils import load_f_encoder, train_classifier_ensemble_for_fs
from data.dataset import sample_support_query


# Single models


def test_few_shot(P, masked_ratio, test_dataset):
    total_acc = 0.0
    total_loss = 0.0

    mode = P.classifier_model
    print(f"Few-shot evaluation mode: {mode}")
    if mode in ["proto", "nn"]:
        print(f"Metric: {P.metric_clf_mode}")

    for step in range(P.test_steps):
        f_model = load_f_encoder(P, masked_ratio)
        support_idxs, query_idxs = sample_support_query(test_dataset, P.shot)

        sup_subset = Subset(test_dataset, support_idxs)
        sup_loader = DataLoader(sup_subset, batch_size=len(support_idxs), shuffle=False)
        x_sup, y_sup = next(iter(sup_loader))
        x_sup, y_sup = x_sup.to(P.device), y_sup.to(P.device)

        qry_subset = Subset(test_dataset, query_idxs)
        qry_loader = DataLoader(qry_subset, batch_size=len(query_idxs), shuffle=False)

        if mode == "probe":
            clf = Classifier(f_model, P.hidden_dim, P.num_classes).to(P.device)
            clf = train_classifier(P, clf, x_sup, y_sup)
            test_loss, test_acc = validate_classifier(P, clf, qry_loader)
        elif mode == "proto":
            test_loss, test_acc = validate_proto(
                P, f_model, x_sup, y_sup, qry_loader, metric=P.metric_clf_mode
            )
        elif mode == "nn":
            test_loss, test_acc = validate_nn(
                P, f_model, x_sup, y_sup, qry_loader, metric=P.metric_clf_mode
            )

        total_acc += test_acc
        total_loss += test_loss

        if step % P.log_interval == 0:
            print(
                f"[Pretrain] Step {step + 1:02d}/{P.test_steps}, current accuracy: {total_acc/(step + 1):.4f} (loss: {total_loss/(step + 1):.4f})"
            )

    print(
        f"[Test] Final Average Accuracy: {total_acc / P.test_steps:.4f} (loss: {total_loss / P.test_steps:.4f})"
    )

    return total_acc / P.test_steps


# Ensemble models


def test_few_shot_ensemble(P, test_dataset):
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    total_loss, total_acc = 0.0, 0.0

    mode = P.classifier_model
    print(f"Few-shot evaluation mode: {mode}")
    if mode in ["proto", "nn"]:
        print(f"Metric: {P.metric_clf_mode}")

    for step in range(P.test_steps):
        fs = [load_f_encoder(P, r) for r in ratios]
        support_idxs, query_idxs = sample_support_query(test_dataset, P.shot)

        sup_subset = Subset(test_dataset, support_idxs)
        sup_loader = DataLoader(sup_subset, batch_size=len(support_idxs), shuffle=False)
        x_sup, y_sup = next(iter(sup_loader))
        x_sup, y_sup = x_sup.to(P.device), y_sup.to(P.device)

        qry_subset = Subset(test_dataset, query_idxs)
        qry_loader = DataLoader(qry_subset, batch_size=len(query_idxs), shuffle=False)

        if mode == "probe":
            clfs = train_classifier_ensemble_for_fs(P, fs, ratios, x_sup, y_sup)
            test_loss, test_acc = validate_classifier_ensemble(P, clfs, qry_loader)
        elif mode == "proto":
            test_loss, test_acc = validate_proto_ensemble(
                P, fs, x_sup, y_sup, qry_loader, metric=P.metric_clf_mode
            )
        elif mode == "nn":
            test_loss, test_acc = validate_nn_ensemble(
                P, fs, x_sup, y_sup, qry_loader, metric=P.metric_clf_mode
            )

        total_loss += test_loss
        total_acc += test_acc

        if step % P.log_interval == 0:
            print(
                f"[Pretrain] Step {step + 1:02d}/{P.test_steps}, current accuracy: {total_acc/(step + 1):.4f} (loss: {total_loss/(step + 1):.4f})"
            )

    print(
        f"[Test] Final Average Accuracy: {total_acc / P.test_steps:.4f} (loss: {total_loss / P.test_steps:.4f})"
    )

    return total_acc / P.test_steps
