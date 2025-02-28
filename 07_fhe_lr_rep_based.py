#!/usr/bin/env python3

"""
borrows heavily from:
https://github.com/zama-ai/concrete-ml/blob/main/docs/advanced_examples/LogisticRegressionTraining.ipynb
"""

import os
import pathlib
import tempfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.linear_model import LogisticRegression as skl_LogisticRegression
from tqdm import tqdm

from concrete.ml.sklearn import SGDClassifier
from concrete.compiler import check_gpu_available
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete import fhe


n_epochs = 10_000
n_train = 289667
batch_size = 8
max_iter = 20
eval_size = 10 * batch_size

"""
load data
"""

if os.uname().nodename.startswith("cri"):
    data_hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data")
else:
    # change following line to develop locally
    data_hm = pathlib.Path(__file__).parent.joinpath("results").absolute()


data_version = "day_stays_qc_first_24h"
model_version = "smaller-lr-search"

splits = ("train", "val", "test")
feats = dict()
mort = dict()

for s in splits:
    feats[s] = np.load(
        data_hm.joinpath(
            f"{data_version}-tokenized", s, "features-{m}.npy".format(m=model_version)
        )
    )
    mort[s] = (
        pl.scan_parquet(
            data_hm.joinpath(f"{data_version}-tokenized", s, "outcomes.parquet")
        )
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .astype(int)
        .ravel()
    )

"""
test a model with scikit-learn
"""

clf = skl_LogisticRegression(random_state=0).fit(feats["train"], mort["train"])
preds = clf.predict_proba(feats["test"])[:, 1]
print("Scikit-learn AUC: {:.3f}".format(skl_auc(y_true=mort["test"], y_score=preds)))
# Scikit-learn AUC: 0.896

"""
preprocess data for quantization
"""

parameters_range = (-1.0, 1.0)
mn, mx = feats["train"].min(axis=0), feats["train"].max(axis=0)
feats["train"] = (2 * feats["train"] - mx - mn) / (mx - mn)
for s in ("val", "test"):
    feats[s] = np.clip(
        (2 * feats[s] - mx - mn) / (mx - mn),
        a_min=min(parameters_range),
        a_max=max(parameters_range),
    )

"""
training on encrypted data
"""

feats["train"] = feats["train"][:n_train, ::]
mort["train"] = mort["train"][:n_train]
d_feat = feats["train"].shape[1]

cML_clf = SGDClassifier(
    random_state=42,
    max_iter=max_iter,
    fit_encrypted=True,
    parameters_range=parameters_range,
)

mn, mx = feats["train"].min(axis=0), feats["train"].max(axis=0)
# y \in {0,1}

# create some dummy data with the same range as the training set
dummy_feats = np.vstack([mn, mx] * (batch_size // 2))
dummy_mort = np.array([0, 1] * (batch_size // 2))

device = "cuda" if check_gpu_available() else "cpu"
print(f"{device=}")
cML_clf.fit(dummy_feats, dummy_mort, fhe="disable", device=device)

cache_loc = pathlib.Path(__file__).parent.joinpath(".tmp").absolute()
cache_loc.mkdir(exist_ok=True)
cache_dir = tempfile.TemporaryDirectory(dir=str(cache_loc))

fhe_dev = FHEModelDev(cache_dir.name, cML_clf)
fhe_dev.save(mode="training")

# On the client
fhe_client = FHEModelClient(cache_dir.name)
fhe_client.load()
serialized_evaluation_keys = fhe_client.get_serialized_evaluation_keys()

# On the server
fhe_server = FHEModelServer(cache_dir.name)
fhe_server.load()


def predict_proba(weights, bias, X):
    linear_model = np.dot(X, weights[0]) + bias[0]
    sigmoid = 1 / (1 + np.exp(-linear_model))
    return sigmoid


def evaluate_auc(weights, bias):
    preds = predict_proba(weights, bias, feats["test"]).squeeze()
    return skl_auc(y_true=mort["test"], y_score=preds)


# random initialization
weights = np.random.rand(1, dummy_feats.shape[1], 1)
bias = np.random.rand(1, 1, 1)
print("Init AUC: {:.3f}".format(evaluate_auc(weights, bias)))


def quantize_encrypt_serialize_batches(fhe_client, x, y, weights, bias, batch_size):
    x_batches_enc, y_batches_enc = [], []

    for i in tqdm(
        range(0, x.shape[0], batch_size), desc="encrypting batches", position=1
    ):

        # Avoid the last batch if it's not a multiple of 'batch_size'
        if i + batch_size < x.shape[0]:
            batch_range = range(i, i + batch_size)
        else:
            break

        # Make the data X (1, batch_size, n_features) and y (1, batch_size, n_targets=1)
        x_batch = np.expand_dims(x[batch_range, :], 0)
        y_batch = np.expand_dims(y[batch_range], (0, 2))

        # Encrypt the batch
        x_batch_enc, y_batch_enc, _, _ = fhe_client.quantize_encrypt_serialize(
            x_batch, y_batch, None, None
        )

        x_batches_enc.append(x_batch_enc)
        y_batches_enc.append(y_batch_enc)

    _, _, weights_enc, bias_enc = fhe_client.quantize_encrypt_serialize(
        None, None, weights, bias
    )

    return x_batches_enc, y_batches_enc, weights_enc, bias_enc


def server_run(
    fhe_server, x_batches_enc, y_batches_enc, weights_enc, bias_enc, evaluation_keys
):

    weights_enc = fhe.Value.deserialize(weights_enc)
    bias_enc = fhe.Value.deserialize(bias_enc)

    evaluation_keys = fhe.EvaluationKeys.deserialize(evaluation_keys)

    # Run the circuit on the server n times, n being the number of batches sent by the user
    for x_batch, y_batch in tqdm(
        zip(x_batches_enc, y_batches_enc), desc="training batches", position=1
    ):
        x_batch = fhe.Value.deserialize(x_batch)
        y_batch = fhe.Value.deserialize(y_batch)

        weights_enc, bias_enc = fhe_server.run(
            (x_batch, y_batch, weights_enc, bias_enc), evaluation_keys
        )

    weights_enc = weights_enc.serialize()
    bias_enc = bias_enc.serialize()

    return weights_enc, bias_enc


for epoch in tqdm(range(n_epochs), desc="training epochs", position=0):

    perm = np.random.permutation(feats["train"].shape[0])[:eval_size]
    x = feats["train"][perm, ::]
    y = mort["train"][perm]

    # Quantize, encrypt and serialize the batched inputs as well as the weight and bias values
    x_batches_enc, y_batches_enc, weights_enc, bias_enc = (
        quantize_encrypt_serialize_batches(fhe_client, x, y, weights, bias, batch_size)
    )

    # Iterate the circuit over the batches on the server
    fitted_weights_enc, fitted_bias_enc = server_run(
        fhe_server,
        x_batches_enc,
        y_batches_enc,
        weights_enc,
        bias_enc,
        serialized_evaluation_keys,
    )

    # Back on the client, deserialize, decrypt and de-quantize the fitted weight and bias values
    weights, bias = fhe_client.deserialize_decrypt_dequantize(
        fitted_weights_enc, fitted_bias_enc
    )

    print("Epoch {} AUC: {:.3f}".format(epoch, evaluate_auc(weights, bias)))


cache_dir.cleanup()
