import random
import torch
import torch.optim as optim
import json
import os
from datetime import datetime

from models import FNetwork, PNetwork
from trainers import pretrainer
from evals import test_few_shot, test_few_shot_ensemble
from data import get_loaders
from common import parse_args, set_seed


def main():
    P = parse_args()
    if P.seed != 0:
        set_seed(P.seed)
        P.index = P.seed
    else:
        P.index = random.randint(1, 1000000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P.device = device

    train_loader, val_loader, test_loader, feature_groups = get_loaders(
        P, P.dataset, P.batch_size, P.seed
    )
    print("Train size:", len(train_loader.dataset))
    print("Val   size:", len(val_loader.dataset))
    print("Test  size:", len(test_loader.dataset))
    f_model = FNetwork(P.input_dim, P.hidden_dim).to(P.device)
    p_model = PNetwork(P.hidden_dim, P.input_dim, P.embed_dim).to(P.device)

    optimizer = optim.Adam(
        list(f_model.parameters()) + list(p_model.parameters()),
        lr=P.learning_rate,
    )

    if P.ensemble:
        print("Ensemble mode")
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        for r in ratios:
            P.masked_ratio = r

            f_model = FNetwork(P.input_dim, P.hidden_dim).to(P.device)
            p_model = PNetwork(P.hidden_dim, P.input_dim, P.embed_dim).to(P.device)

            optimizer = optim.Adam(
                list(f_model.parameters()) + list(p_model.parameters()),
                lr=P.learning_rate,
            )

            print(f"[Pretrain] masked_ratio={r}, filling mode={P.fill_mode}")
            pretrainer(
                P, f_model, p_model, optimizer, train_loader, val_loader, feature_groups
            )

            del f_model
            del p_model
            del optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        results = {
            "dataset": P.dataset,
            "ensemble": True,
            "configs": [
                {"masked_ratio": r} for r in ratios
            ],
            "fill_mode": P.fill_mode,
            "hidden_dim": P.hidden_dim,
            "embed_dim": P.embed_dim,
            "temperature": P.temperature,
            "seed": P.seed,
            "index": P.index,
            "pretrainer_epochs": P.pretrainer_epochs,
            "learning_rate": P.learning_rate,
            "batch_size": P.batch_size,
            "test_steps": P.test_steps,
            "timestamp": datetime.now().isoformat(),
            "shots": {}
        }

        for shot in [1, 5, 10]:
            P.shot = shot
            
            if shot == 1:
                P.classifier_model = "proto"
                P.metric_clf_mode = "cosine"
            else:
                P.classifier_model = "probe"
                P.metric_clf_mode = "euclidean"
            
            print(f"[Evaluating ensemble with {shot}-shot, classifier={P.classifier_model}, metric={P.metric_clf_mode if shot == 1 else 'N/A'}]")
            avg_accuracy = test_few_shot_ensemble(P, test_loader.dataset)
            results["shots"][shot] = {
                "accuracy": avg_accuracy,
                "classifier_model": P.classifier_model,
                "metric_clf_mode": P.metric_clf_mode if shot == 1 else None
            }
            print(f"{shot}-shot ensemble accuracy: {avg_accuracy:.4f}")

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir, 
            f"{P.dataset}_ensemble_{P.index}_results.json"
        )
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")
    else:
        print("Single model mode")
        print(f"[Pretrain] masked_ratio={P.masked_ratio}, filling mode={P.fill_mode}")
        pretrainer(
            P, f_model, p_model, optimizer, train_loader, val_loader, feature_groups
        )

        results = {
            "dataset": P.dataset,
            "masked_ratio": P.masked_ratio,
            "fill_mode": P.fill_mode,
            "hidden_dim": P.hidden_dim,
            "embed_dim": P.embed_dim,
            "temperature": P.temperature,
            "classifier_model": P.classifier_model,
            "metric_clf_mode": P.metric_clf_mode,
            "seed": P.seed,
            "index": P.index,
            "pretrainer_epochs": P.pretrainer_epochs,
            "learning_rate": P.learning_rate,
            "batch_size": P.batch_size,
            "test_steps": P.test_steps,
            "timestamp": datetime.now().isoformat(),
            "shots": {}
        }

        for shot in [1, 5, 10]:
            P.shot = shot
            
            if shot == 1:
                P.classifier_model = "proto"
                P.metric_clf_mode = "cosine"
            else:
                P.classifier_model = "probe"
            
            print(f"[Evaluating with {shot}-shot, classifier={P.classifier_model}, metric={P.metric_clf_mode if shot == 1 else 'N/A'}]")
            avg_accuracy = test_few_shot(P, P.masked_ratio, test_loader.dataset)
            results["shots"][shot] = {
                "accuracy": avg_accuracy,
                "classifier_model": P.classifier_model,
                "metric_clf_mode": P.metric_clf_mode if shot == 1 else None
            }
            print(f"{shot}-shot accuracy: {avg_accuracy:.4f}")

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir, 
            f"{P.dataset}_{P.masked_ratio}_{P.index}_results.json"
        )
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
