"""Fiduccia-Mattheyses (FM) refinement of the per-strand 2-way mask
(mask_updating_test_3D.ipynb, FM cell).

Neither mode builds an (S, S) adjacency; only the cell <-> strand incidence is used:
  mode="clique": each cell is a clique, edge weight = number of cells two strands share;
                 minimizes sum_c n_true[c] * n_false[c]
  mode="cutnet": each cell is a net (hypergraph); minimizes the number of cells containing both
                 colors (the cut-net objective of the original FM paper)
  lam: extra deviation penalty, gain += -lam * (state != state0) with state0 = the state passed to
       fm_refine; larger lam keeps strands closer to the original mask (default 0 = no limit)
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse


def build_cell_incidence(strand_mask_grid):
    """(n_rows, n_cols, S) bool -> 只保留含 >= 2 根髮絲的格子 (單根的格子不可能混色，對兩種目標都沒貢獻)，
    回傳 (cell_ptr, cell_members, strand_ptr, strand_cells) 兩份 CSR 索引：格子 -> 髮絲 / 髮絲 -> 格子。
    記憶體只有 O(nnz)，nnz = 所有 (格子, 髮絲) 配對數。"""
    n_rows, n_cols, n_strands = strand_mask_grid.shape
    M = sparse.csr_matrix(strand_mask_grid.reshape(n_rows * n_cols, n_strands))
    return _incidence_from_csr(M)


def _incidence_from_csr(M):
    M = M[np.diff(M.indptr) >= 2].tocsr()
    Mt = M.T.tocsr()
    return (M.indptr.astype(np.int64), M.indices.astype(np.int32),
            Mt.indptr.astype(np.int64), Mt.indices.astype(np.int32))


@njit
def _b_insert(i, key, head, nxt, prv):
    h = head[key]
    nxt[i] = h
    prv[i] = -1
    if h >= 0:
        prv[h] = i
    head[key] = i


@njit
def _b_remove(i, key, head, nxt, prv):
    p = prv[i]
    n = nxt[i]
    if p >= 0:
        nxt[p] = n
    else:
        head[key] = n
    if n >= 0:
        prv[n] = p


@njit
def _bump(u, d, gain, off, W, state, head, nxt, prv, tops):
    s = 1 if state[u] else 0  # u 還沒被鎖定，pass 期間 state[u] 不變，所以 bucket 側別固定
    _b_remove(u, s * W + gain[u] + off, head, nxt, prv)
    gain[u] += d
    key = gain[u] + off
    _b_insert(u, s * W + key, head, nxt, prv)
    if key > tops[s]:
        tops[s] = key


@njit
def _fm_pass(state, cell_ptr, cell_members, strand_ptr, strand_cells, mode, max_gain, patience, lo, hi, state0, lam):
    """一個 FM pass (in-place 修改 state)，回傳這個 pass 實際採用的累積 gain。
    mode 0 = clique, 1 = cutnet。gain bucket 是 (head, nxt, prv) 三個 int 陣列做成的雙向鏈結串列，
    True / False 兩側各一組 bucket (head 的前後兩半)，這樣才能在 True 的總數被限制在 [lo, hi]
    時，只從「移動後仍合法」的那一側挑最大 gain，避免整個分類塌縮成同一類。
    state0/lam: 偏移懲罰，只影響初始 gain (每根髮絲自己是否偏離 state0，跟鄰居無關，pass 中不必增量更新)。"""
    S = state.shape[0]
    R = cell_ptr.shape[0] - 1
    n1 = np.zeros(R, np.int64)  # 每格目前 True 的髮絲數
    size = np.empty(R, np.int64)
    for c in range(R):
        size[c] = cell_ptr[c + 1] - cell_ptr[c]
        for k in range(cell_ptr[c], cell_ptr[c + 1]):
            if state[cell_members[k]]:
                n1[c] += 1

    gain = np.zeros(S, np.int64)
    for v in range(S):
        g = 0
        for k in range(strand_ptr[v], strand_ptr[v + 1]):
            c = strand_cells[k]
            nF = n1[c] if state[v] else size[c] - n1[c]  # v 這一側 (含 v) 的髮絲數
            nT = size[c] - nF
            if mode == 0:
                g += nT - (nF - 1)
            else:
                if nF == 1:
                    g += 1
                if nT == 0:
                    g -= 1
        gain[v] = g + (lam if state[v] != state0[v] else -lam)  # 已偏離 state0 -> 移回去 +lam；還沒偏離 -> 移走 -lam

    off = max_gain
    W = 2 * max_gain + 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(S, np.int64)
    prv = np.empty(S, np.int64)
    n_true = 0
    for v in range(S):
        sd = 1 if state[v] else 0
        n_true += sd
        _b_insert(v, sd * W + gain[v] + off, head, nxt, prv)

    locked = np.zeros(S, np.bool_)
    moves = np.empty(S, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0
    tops = np.full(2, 2 * max_gain, np.int64)

    while n_moves < S:
        best_key = -1
        best_side = -1
        for sd in range(2):
            if sd == 1 and n_true - 1 < lo:  # 移動 True 側會讓 True 總數低於下限
                continue
            if sd == 0 and n_true + 1 > hi:
                continue
            while tops[sd] >= 0 and head[sd * W + tops[sd]] < 0:
                tops[sd] -= 1
            if tops[sd] > best_key:
                best_key = tops[sd]
                best_side = sd
        if best_side < 0:
            break
        v = head[best_side * W + best_key]
        _b_remove(v, best_side * W + best_key, head, nxt, prv)
        locked[v] = True
        cum += gain[v]
        moves[n_moves] = v
        n_moves += 1
        n_true += -1 if best_side == 1 else 1
        if cum > best:
            best = cum
            best_step = n_moves

        sv = state[v]
        for k in range(strand_ptr[v], strand_ptr[v + 1]):
            c = strand_cells[k]
            a = cell_ptr[c]
            b = cell_ptr[c + 1]
            nF = n1[c] if sv else size[c] - n1[c]
            nT = size[c] - nF
            if mode == 0:
                for m in range(a, b):
                    u = cell_members[m]
                    if u == v or locked[u]:
                        continue
                    _bump(u, 2 if state[u] == sv else -2, gain, off, W, state, head, nxt, prv, tops)
            else:
                if nT == 0:  # 整格都在 v 這側 -> 移動 v 會讓其他人「離開 v 這側」變成有機會消除混色
                    for m in range(a, b):
                        u = cell_members[m]
                        if u != v and not locked[u]:
                            _bump(u, 1, gain, off, W, state, head, nxt, prv, tops)
                elif nT == 1:  # 對側唯一的那根，之後移過來就不再消除混色
                    for m in range(a, b):
                        u = cell_members[m]
                        if state[u] != sv:
                            if not locked[u]:
                                _bump(u, -1, gain, off, W, state, head, nxt, prv, tops)
                            break
            if sv:
                n1[c] -= 1
            else:
                n1[c] += 1
            if mode == 1:
                nF2 = nF - 1
                if nF2 == 0:  # v 走後原本這側空了 -> 其他人 (都在對側) 移過去不再有消除混色的機會
                    for m in range(a, b):
                        u = cell_members[m]
                        if u != v and not locked[u]:
                            _bump(u, -1, gain, off, W, state, head, nxt, prv, tops)
                elif nF2 == 1:  # 原本這側只剩一根 -> 它移走就能消除混色
                    for m in range(a, b):
                        u = cell_members[m]
                        if u != v and state[u] == sv:
                            if not locked[u]:
                                _bump(u, 1, gain, off, W, state, head, nxt, prv, tops)
                            break
        state[v] = not sv

        if n_moves - best_step >= patience:  # 連續 patience 步累積 gain 沒創新高，提前結束這個 pass
            break

    for i in range(best_step, n_moves):  # 回滾沒被最佳前綴採用的移動
        state[moves[i]] = not state[moves[i]]
    return best


def fm_objectives(incidence, state):
    """回傳 (混色格數 cut-net, clique cut = sum_c n_true * n_false)。"""
    cell_ptr, cell_members, _, _ = incidence
    n1 = np.add.reduceat(state[cell_members].astype(np.int64), cell_ptr[:-1])
    size = np.diff(cell_ptr)
    return int(((n1 > 0) & (n1 < size)).sum()), int((n1 * (size - n1)).sum())


def fm_refine(incidence, state, mode="cutnet", max_passes=20, patience=None, balance_tol=0.02, lam=0, verbose=True, return_history=False):
    """FM 迭代更新 2-way 分類 state (bool, (S,))，回傳新的 state (不改動輸入)。
    每個 pass 依 gain 由大到小把髮絲各移動一次並鎖住，只採用累積 gain 最大的前綴，重複到沒有改善。
    patience: 連續多少步累積 gain 沒創新高就提前結束該 pass；None = 不提前結束 (完整 FM)。
    balance_tol: True 的總數只能偏離初始值 balance_tol * S 以內 (預設 2%)；None = 不限制。
      cut-net 目標「全部同一類」就是 0 混色，不設限制的話會塌縮成單一顏色，所以預設要限。
    lam: 偏移懲罰係數 (非負整數，預設 0 = 不限制)。gain 額外加上 -lam * (state != state0)，
      state0 = 傳入 fm_refine 當下的初始 state (整個 fm_refine 過程固定，不會每個 pass 重設)，
      也就是每根髮絲想離開自己的原始顏色都要先「還」lam 的 gain，藉此壓低跟原始 mask 的偏移量。
    return_history: True 時回傳 [初始 state, pass 0 後, pass 1 後, ...] 的 list。"""
    cell_ptr, cell_members, strand_ptr, strand_cells = incidence
    n_strands = strand_ptr.shape[0] - 1
    assert state.shape == (n_strands,)
    state = state.astype(np.bool_).copy()
    state0 = state.copy()  # lam 懲罰的參照點，整個 fm_refine 期間固定
    lam = int(lam)
    assert lam >= 0, "lam 必須是非負整數"

    if mode == "clique":
        mode_id = 0
        entry_strand = np.repeat(np.arange(n_strands), np.diff(strand_ptr))
        max_gain = int(np.bincount(entry_strand, weights=np.diff(cell_ptr)[strand_cells] - 1,
                                   minlength=n_strands).max())
    elif mode == "cutnet":
        mode_id = 1
        max_gain = int(np.diff(strand_ptr).max())
    else:
        raise ValueError(mode)
    max_gain = max(max_gain, 1) + lam  # gain 多了 ±lam 的偏移懲罰，bucket 範圍要跟著放大
    if patience is None:
        patience = n_strands + 1
    n_true0 = int(state.sum())
    tol = n_strands if balance_tol is None else int(balance_tol * n_strands)
    lo, hi = max(0, n_true0 - tol), min(n_strands, n_true0 + tol)

    history = [state.copy()] if return_history else None

    if verbose:
        mixed, cut = fm_objectives(incidence, state)
        print(f"[FM/{mode}] start: mixed cells = {mixed}, clique cut = {cut}, lam = {lam}")
    for pass_idx in range(max_passes):
        gained = _fm_pass(state, cell_ptr, cell_members, strand_ptr, strand_cells,
                          mode_id, max_gain, patience, lo, hi, state0, lam)
        if verbose:
            mixed, cut = fm_objectives(incidence, state)
            print(f"[FM/{mode}] pass {pass_idx}: gain = {gained}, mixed cells = {mixed}, "
                  f"clique cut = {cut}, diff from state0 = {(state != state0).mean():.4f}")
        if return_history:
            history.append(state.copy())
        if gained <= 0:
            break

    return history if return_history else state
