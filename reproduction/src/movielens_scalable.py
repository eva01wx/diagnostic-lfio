"""Sparse, temporal MovieLens study with exact additive item-rule explanations.

All eligible training interactions are used. The rule vocabulary is bounded by
training-popular anchor movies; evaluation ranks the complete training catalogue
for every eligible user, in batches. This is an application study, not SANSA.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from src.movielens_experiment import resolve_movielens_path


def temporal_state(u, item, rating, timestamp, cutoff, stop, n_users, n_items, min_count=5):
    train_time = timestamp < cutoff
    user_counts = np.bincount(u[train_time], minlength=n_users)
    item_counts = np.bincount(item[train_time], minlength=n_items)
    known_user = user_counts >= min_count
    catalogue = np.flatnonzero(item_counts >= min_count)
    mapping = np.full(n_items, -1, dtype=np.int32)
    mapping[catalogue] = np.arange(len(catalogue), dtype=np.int32)
    train = train_time & known_user[u] & (mapping[item] >= 0)
    rows, cols = u[train], mapping[item[train]]
    seen = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, len(catalogue)))
    seen.data[:] = 1
    positive = train & (rating >= 4)
    X = sparse.csr_matrix((np.ones(int(positive.sum()), np.float32),
                           (u[positive], mapping[item[positive]])), shape=seen.shape)
    X.data[:] = 1
    heldout_time = (timestamp >= cutoff) & (timestamp < stop)
    heldout = heldout_time & known_user[u] & (mapping[item] >= 0) & (rating >= 4)
    truth = sparse.csr_matrix((np.ones(int(heldout.sum()), np.float32),
                               (u[heldout], mapping[item[heldout]])), shape=seen.shape)
    truth.data[:] = 1
    seen_positives = truth.multiply(seen)
    truth = truth - seen_positives
    truth.eliminate_zeros()
    users = np.flatnonzero(np.diff(truth.indptr) > 0)
    metadata = {'cutoff_timestamp': int(cutoff), 'stop_timestamp': int(stop),
                'train_ratings': int(train.sum()), 'train_positive_pairs': X.nnz,
                'train_users': int(known_user.sum()), 'catalogue_items': len(catalogue),
                'raw_heldout_ratings': int(heldout_time.sum()),
                'heldout_unknown_user_ratings': int((heldout_time & ~known_user[u]).sum()),
                'heldout_unknown_item_for_known_user': int((heldout_time & known_user[u] & (mapping[item] < 0)).sum()),
                'eligible_users': len(users), 'heldout_positive_pairs': truth.nnz,
                'seen_positive_pairs_excluded': seen_positives.nnz}
    return X, seen, truth, users, catalogue, metadata


def anchor_statistics(X, n_anchors):
    frequency = np.asarray(X.sum(axis=0)).ravel()
    anchors = np.argsort(-frequency, kind='stable')[:min(n_anchors, len(frequency))]
    counts = (X[:, anchors].T @ X).toarray().astype(np.float32, copy=False)
    counts[np.arange(len(anchors)), anchors] = 0
    return anchors, frequency, counts


def item_rule_weights(counts, frequency, anchors, n_users, method, neighbors, shrink=20):
    if method == 'cosine':
        weights = counts / np.sqrt(np.maximum(frequency[anchors, None] * frequency[None, :], 1))
    elif method == 'shrunk_lift':
        expected = frequency[anchors, None] * frequency[None, :] / max(1, n_users)
        weights = np.log1p(counts / np.maximum(expected, 1e-8)) * counts / (counts + shrink)
    else:
        raise ValueError(method)
    weights[counts < 3] = 0
    k = min(neighbors, weights.shape[1])
    columns = np.argsort(-weights, axis=1, kind='stable')[:, :k]
    values = np.take_along_axis(weights, columns, axis=1)
    rows = np.repeat(np.arange(len(anchors)), k)
    result = sparse.csr_matrix((values.ravel(), (rows, columns.ravel())), shape=weights.shape)
    result.eliminate_zeros()
    return result


def exact_top_k(scores, k):
    """Resolve score ties by catalogue index, including ties at the K boundary."""
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) <= k:
        return finite[np.lexsort((finite, -scores[finite]))]
    threshold = np.partition(scores, len(scores) - k)[-k]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[:k - len(greater)]
    candidates = np.concatenate((greater, equal))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def evaluate(X, seen, truth, users, frequency, anchors=None, weights=None, k=10, batch_size=128):
    if not len(users):
        raise ValueError('No users have unseen positive holdout items in the training catalogue')
    popularity = frequency / max(1, frequency.max())
    discount = 1 / np.log2(np.arange(k) + 2)
    ideal = np.r_[0, np.cumsum(discount)]
    rows, recommended = [], set()
    for first in range(0, len(users), batch_size):
        batch = users[first:first + batch_size]
        if weights is None:
            scores = np.broadcast_to(popularity, (len(batch), len(popularity))).copy()
            supported = None
        else:
            scores = (X[batch][:, anchors] @ weights).toarray()
            supported = scores > 0
            # Popularity is an explicit tie-break/fallback, not a rule explanation.
            scores += 1e-6 * popularity
        for offset, uid in enumerate(batch):
            history = seen.indices[seen.indptr[uid]:seen.indptr[uid + 1]]
            scores[offset, history] = -np.inf
            rec = exact_top_k(scores[offset], k)
            relevant = truth.indices[truth.indptr[uid]:truth.indptr[uid + 1]]
            hit = np.isin(rec, relevant)
            hits = int(hit.sum())
            explained = int(supported[offset, rec].sum()) if supported is not None else 0
            rows.append({'user_id': int(uid + 1), 'precision': hits / k,
                         'recall': hits / len(relevant), 'ndcg': float((hit * discount[:len(rec)]).sum() / ideal[min(k, len(relevant))]),
                         'hits': hits, 'relevant': len(relevant), 'recommended': len(rec),
                         'rule_explained': explained})
            recommended.update(rec.tolist())
        if first and first % (batch_size * 100) == 0:
            print(f'  evaluated {first:,}/{len(users):,} users', flush=True)
    metrics = {key: float(np.mean([r[key] for r in rows])) for key in ('precision', 'recall', 'ndcg')}
    metrics.update(num_users=len(rows), hit_rate=float(np.mean([r['hits'] > 0 for r in rows])),
                   catalogue_coverage=len(recommended) / len(frequency),
                   explanation_coverage=sum(r['rule_explained'] for r in rows) / max(1, sum(r['recommended'] for r in rows)))
    return metrics, rows


def example_explanations(X, seen, users, frequency, anchors, weights, movie_ids, titles):
    examples = []
    for uid in users[:3]:
        active = X[uid, anchors].indices
        rule_scores = np.asarray(weights[active].sum(axis=0)).ravel()
        fallback = 1e-6 * frequency / max(1, frequency.max())
        scores = rule_scores + fallback
        scores[seen[uid].indices] = -np.inf
        for item in exact_top_k(scores, 3):
            contributions = [(int(anchors[a]), float(weights[a, item])) for a in active if weights[a, item] > 0]
            examples.append({'user_id':int(uid + 1), 'movie_id':int(movie_ids[item]),
                             'title':titles[int(movie_ids[item])], 'score':float(scores[item]),
                             'popularity_component':float(fallback[item]),
                             'rule_contribution_sum':float(sum(w for _, w in contributions)),
                             'contributions':[{'because_liked_movie_id':int(movie_ids[a]),
                                               'title':titles[int(movie_ids[a])], 'weight':w}
                                              for a, w in sorted(contributions, key=lambda c:-c[1])]})
    return examples


def run(data_path, output_dir, n_anchors=128, neighbors=(50, 200), shrinkages=(10, 50), batch_size=128):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(resolve_movielens_path(data_path))
    started = time.perf_counter()
    print(f'Loading all ratings from {root}', flush=True)
    movies = pd.read_csv(root / 'movies.csv').sort_values('movieId')
    movie_ids = movies.movieId.to_numpy()
    frame = pd.read_csv(root / 'ratings.csv', dtype={'userId':'int32', 'movieId':'int32', 'rating':'float32', 'timestamp':'int64'})
    u = frame.userId.to_numpy() - 1
    item = np.searchsorted(movie_ids, frame.movieId.to_numpy()).astype(np.int32)
    rating, timestamp = frame.rating.to_numpy(), frame.timestamp.to_numpy()
    n_users = int(u.max()) + 1
    cut60, cut80 = np.quantile(timestamp, [0.6, 0.8], method='higher').astype(np.int64)
    if cut60 >= cut80:
        raise ValueError('Timestamp ties leave an empty validation period')
    total = len(frame)
    del frame
    trace = []
    selected = {}
    state = temporal_state(u, item, rating, timestamp, cut60, cut80, n_users, len(movies))
    X, seen, truth, users, catalogue, validation_protocol = state
    print('Validation protocol:', json.dumps(validation_protocol), flush=True)
    anchors, frequency, counts = anchor_statistics(X, n_anchors)
    pop_metrics, _ = evaluate(X, seen, truth, users, frequency, batch_size=batch_size)
    trace.append({'method':'popularity', 'metrics':pop_metrics})
    for method in ('cosine', 'shrunk_lift'):
        trials = []
        for neighbors_ in neighbors:
            for shrink in ((0,) if method == 'cosine' else shrinkages):
                weights = item_rule_weights(counts, frequency, anchors, validation_protocol['train_users'], method, neighbors_, shrink)
                metrics, _ = evaluate(X, seen, truth, users, frequency, anchors, weights, batch_size=batch_size)
                trial = {'method':method, 'neighbors':neighbors_, 'shrink':shrink, 'rules':weights.nnz, 'metrics':metrics}
                trials.append(trial)
                trace.append(trial)
                print('Validation:', json.dumps(trial), flush=True)
        selected[method] = max(trials, key=lambda t:(t['metrics']['ndcg'], -t['rules']))
    del state, X, seen, truth, counts
    # Hyperparameters are frozen before refitting on training + validation.
    state = temporal_state(u, item, rating, timestamp, cut80, int(timestamp.max()) + 1, n_users, len(movies))
    X, seen, truth, users, catalogue, test_protocol = state
    print('Final protocol:', json.dumps(test_protocol), flush=True)
    anchors, frequency, counts = anchor_statistics(X, n_anchors)
    results = {}
    for method in ('popularity', 'cosine', 'shrunk_lift'):
        if method == 'popularity':
            weights = None
        else:
            config = selected[method]
            weights = item_rule_weights(counts, frequency, anchors, test_protocol['train_users'], method, config['neighbors'], config['shrink'])
            sparse.save_npz(output / f'{method}_rules.npz', weights)
            examples = example_explanations(X, seen, users, frequency, anchors, weights,
                                             movie_ids[catalogue], dict(zip(movies.movieId, movies.title)))
            (output / f'{method}_explanations.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
        metrics, rows = evaluate(X, seen, truth, users, frequency, anchors, weights, batch_size=batch_size)
        metrics['rules'] = weights.nnz if weights is not None else 0
        results[method] = metrics
        pd.DataFrame(rows).to_csv(output / f'{method}_per_user.csv', index=False)
        print('Test:', method, json.dumps(metrics), flush=True)
    np.save(output / 'catalogue_movie_ids.npy', movie_ids[catalogue])
    np.save(output / 'anchor_movie_ids.npy', movie_ids[catalogue[anchors]])
    result = {'protocol':'temporal_60_20_20_refit_no_test_tuning', 'raw_ratings':total,
              'data_path':str(root), 'n_anchors':len(anchors), 'all_eligible_users_evaluated':True,
              'training_interaction_sampling':False, 'candidate_user_sampling':False,
              'vocabulary_note':'Rules use training-popular anchor items; all training catalogue items remain ranking candidates.',
              'validation_protocol':validation_protocol, 'test_protocol':test_protocol,
              'validation_trials':trace, 'selected':selected, 'methods':results,
              'elapsed_seconds':time.perf_counter()-started,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python':platform.python_version(),
              'completed_utc':datetime.now(timezone.utc).isoformat()}
    (output / 'movielens_scalable.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='ml-latest')
    parser.add_argument('--output-dir', default='results/movielens_scalable')
    parser.add_argument('--anchors', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    run(args.data_path, args.output_dir, n_anchors=args.anchors, batch_size=args.batch_size)
