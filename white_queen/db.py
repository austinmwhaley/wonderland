"""SQLite interface. Two tables + coverage-diet views. The DB is the toy event
stream: episodes = sessions, transitions = events, policies = behavior actors.

Observations are stored as explicit REAL columns sized to obs_dim (see
meta.obs_dim), so the store is observation-dimension agnostic (CartPole 4,
MountainCar 2, Acrobot 6, ...). Behavior probs are exactly reconstructible
for discrete actions from (greedy_action, eps):
mu[greedy] = 1-eps+eps/nA else eps/nA. No estimation anywhere in Stage 1 —
the lab analog of exact propensity logs.
"""
import sqlite3

import numpy as np

_POLICIES_AND_EPISODES = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS policies (
    policy_id   TEXT PRIMARY KEY,
    family      TEXT NOT NULL,   -- value-based | policy-gradient | random
    algo        TEXT NOT NULL,   -- lab class name or 'random'
    checkpoint_frac REAL,        -- NULL for random; else fraction of training
    seed        INTEGER NOT NULL,
    train_steps INTEGER NOT NULL,
    collect_eps REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id  TEXT NOT NULL REFERENCES policies(policy_id),
    seed       INTEGER NOT NULL,
    ret        REAL NOT NULL,
    length     INTEGER NOT NULL
);
"""


def _schema(obs_dim):
    ocols = ", ".join(f"o{i} REAL" for i in range(obs_dim))
    ncols = ", ".join(f"n{i} REAL" for i in range(obs_dim))
    return (
        _POLICIES_AND_EPISODES
        + f"""CREATE TABLE IF NOT EXISTS transitions (
    tid INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(episode_id),
    t INTEGER NOT NULL,
    {ocols},
    action INTEGER NOT NULL,
    reward REAL NOT NULL,
    {ncols},
    done REAL NOT NULL,
    greedy_action INTEGER NOT NULL,
    eps REAL NOT NULL,
    prob_taken REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tr_ep ON transitions(episode_id);
CREATE INDEX IF NOT EXISTS idx_ep_pol ON episodes(policy_id);
"""
    )

# Coverage diets: same rows, different lenses. novice_only ~= thin/poor coverage
# stress; expert_only ~= imitation-ceiling check; low_coverage ~= sparse support.
VIEWS = {
    "mixed": "SELECT t.* FROM transitions t",
    "novice_only": (
        "SELECT t.* FROM transitions t JOIN episodes e ON t.episode_id=e.episode_id "
        "JOIN policies p ON e.policy_id=p.policy_id "
        "WHERE p.algo='random' OR IFNULL(p.checkpoint_frac,1.0) <= 0.06"
    ),
    "expert_only": (
        "SELECT t.* FROM transitions t JOIN episodes e ON t.episode_id=e.episode_id "
        "JOIN policies p ON e.policy_id=p.policy_id "
        "WHERE IFNULL(p.checkpoint_frac,0.0) >= 1.0"
    ),
    "low_coverage": "SELECT t.* FROM transitions t WHERE t.tid % 4 = 0",
}


def connect(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_db(path, obs_dim=4, nA=2, gamma=0.99):
    con = connect(path)
    con.executescript(_schema(int(obs_dim)))
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                    [("obs_dim", str(obs_dim)), ("nA", str(nA)), ("gamma", str(gamma))])
    for name, sql in VIEWS.items():
        con.execute(f"DROP VIEW IF EXISTS diet_{name}")
        con.execute(f"CREATE VIEW diet_{name} AS {sql}")
    con.commit()
    return con


def register_policy(con, policy_id, family, algo, checkpoint_frac, seed, train_steps, collect_eps):
    con.execute(
        "INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?,?,?)",
        (policy_id, family, algo, checkpoint_frac, seed, train_steps, collect_eps))
    con.commit()


def insert_episode(con, policy_id, seed, obs_seq, act_seq, rew_seq, done_seq,
                   greedy_seq, eps_seq, prob_seq):
    cur = con.execute(
        "INSERT INTO episodes (policy_id, seed, ret, length) VALUES (?,?,?,?)",
        (policy_id, seed, float(np.sum(rew_seq)), len(act_seq)))
    eid = cur.lastrowid
    obs2 = np.vstack([obs_seq[1:], obs_seq[-1:]])  # next-obs implicit in rollout order
    d = int(np.asarray(obs_seq).shape[1])
    ocols = ",".join(f"o{i}" for i in range(d))
    ncols = ",".join(f"n{i}" for i in range(d))
    ph = ",".join(["?"] * (2 * d + 8))
    rows = [(eid, t,
             *map(float, obs_seq[t]), int(act_seq[t]), float(rew_seq[t]),
             *map(float, obs2[t]), float(done_seq[t]),
             int(greedy_seq[t]), float(eps_seq[t]), float(prob_seq[t]))
            for t in range(len(act_seq))]
    con.executemany(
        f"INSERT INTO transitions (episode_id,t,{ocols},action,reward,"
        f"{ncols},done,greedy_action,eps,prob_taken) VALUES ({ph})", rows)
    con.commit()
    return eid


def load_diet(path, diet):
    """Returns dict of float32/int64 arrays + exact behavior probs mu (N,nA).

    Fast path uses DuckDB over the SQLite file with Arrow/zero-copy fetch and
    computes the step-within-episode t via a window function (vectorized),
    replacing the old Python per-row loop. Pandas is never used. Falls back
    to sqlite3 if DuckDB is unavailable.
    """
    try:
        import duckdb
        con = duckdb.connect()
        con.execute(f"ATTACH '{path}' AS wq (TYPE sqlite, READ_ONLY)")
        nA = int(con.execute(
            "SELECT value FROM wq.meta WHERE key='nA'").fetchone()[0])
        d = int(con.execute(
            "SELECT value FROM wq.meta WHERE key='obs_dim'").fetchone()[0])
        oc = ",".join(f"o{i}" for i in range(d))
        nc = ",".join(f"n{i}" for i in range(d))
        q = (
            f"SELECT {oc},action,reward,{nc},done,"
            f"greedy_action,eps,prob_taken,episode_id,"
            f"ROW_NUMBER() OVER (PARTITION BY episode_id ORDER BY tid) - 1 AS t "
            f"FROM wq.diet_{diet} ORDER BY tid")
        cols = con.execute(q).fetchnumpy()  # numpy arrays via Arrow
        con.close()
    except Exception:
        return _load_diet_sqlite(path, diet)
    N = len(cols["action"])
    if N == 0:
        raise ValueError(f"diet '{diet}' is empty — run the colony first")
    obs = np.stack([cols[f"o{i}"] for i in range(d)], 1).astype(np.float32)
    obs2 = np.stack([cols[f"n{i}"] for i in range(d)], 1).astype(np.float32)
    act = np.asarray(cols["action"], dtype=np.int64)
    rew = np.asarray(cols["reward"], dtype=np.float32)
    done = np.asarray(cols["done"], dtype=np.float32)
    greedy = np.asarray(cols["greedy_action"], dtype=np.int64)
    eps = np.asarray(cols["eps"], dtype=np.float32)
    ep = np.asarray(cols["episode_id"], dtype=np.int64)
    t = np.asarray(cols["t"], dtype=np.int64)
    mu = np.full((N, nA), 0.0, dtype=np.float32)
    mu[np.arange(N), greedy] = 1.0 - eps
    mu += (eps / nA)[:, None]
    mu_take = mu[np.arange(N), act]
    return {"obs": obs, "act": act, "rew": rew, "obs2": obs2, "done": done,
            "mu": mu, "mu_take": mu_take, "episode": ep, "t": t,
            "nA": nA, "N": N, "mode": "rl"}


def _load_diet_sqlite(path, diet):
    """sqlite3 fallback (identical result; vectorized t via numpy diff)."""
    con = connect(path)
    nA = int(con.execute("SELECT value FROM meta WHERE key='nA'").fetchone()[0])
    d = int(con.execute("SELECT value FROM meta WHERE key='obs_dim'").fetchone()[0])
    oc = ",".join(f"o{i}" for i in range(d))
    nc = ",".join(f"n{i}" for i in range(d))
    rows = con.execute(
        f"SELECT {oc},action,reward,{nc},done,"
        f"greedy_action,eps,prob_taken,episode_id FROM diet_{diet} ORDER BY tid").fetchall()
    con.close()
    a = np.asarray(rows, dtype=np.float64)
    if len(a) == 0:
        raise ValueError(f"diet '{diet}' is empty — run the colony first")
    obs = a[:, 0:d].astype(np.float32)
    act = a[:, d].astype(np.int64)
    rew = a[:, d + 1].astype(np.float32)
    obs2 = a[:, d + 2:d + 2 + d].astype(np.float32)
    done = a[:, d + 2 + d].astype(np.float32)
    greedy = a[:, d + 3 + d].astype(np.int64)
    eps = a[:, d + 4 + d].astype(np.float32)
    ep = a[:, d + 6 + d].astype(np.int64)
    mu = np.full((len(a), nA), (eps / nA)[:, None], dtype=np.float32)
    mu[np.arange(len(a)), greedy] += 1.0 - eps
    change = np.flatnonzero(np.diff(ep) != 0) + 1
    starts = np.concatenate([[0], change])
    seg_id = np.cumsum(np.concatenate([[0], np.diff(ep) != 0]))
    t = (np.arange(len(ep)) - starts[seg_id]).astype(np.int64)
    mu_take = mu[np.arange(len(a)), act]
    return {"obs": obs, "act": act, "rew": rew, "obs2": obs2, "done": done,
            "mu": mu, "mu_take": mu_take, "episode": ep, "t": t,
            "nA": nA, "N": len(a), "mode": "rl"}


def diet_stats(path):
    con = connect(path)
    out = {}
    for name in VIEWS:
        rows = con.execute(
            f"SELECT e.episode_id, AVG(e.ret) FROM diet_{name} t "
            f"JOIN episodes e ON t.episode_id=e.episode_id "
            f"GROUP BY e.episode_id").fetchall()
        n_eps = len(rows)
        n_tr = con.execute(f"SELECT COUNT(*) FROM diet_{name}").fetchone()[0]
        mean_ret = float(sum(r[1] for r in rows) / n_eps) if n_eps else 0.0
        out[name] = {"transitions": n_tr, "episodes": n_eps,
                     "mean_ep_return": mean_ret}
    con.close()
    return out
