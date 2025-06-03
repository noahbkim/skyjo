import numpy as np

import skyjo as sj

# MARK: Probability Helpers


class DiscreteProbabilityDistribution:
    def __init__(
        self,
        probabilities: np.ndarray[tuple[int], np.float32],
        values: np.ndarray[tuple[int], np.float32],
    ) -> float:
        assert probabilities.shape == values.shape, (
            f"Probabilities and values must have the same shape, got {probabilities.shape} and {values.shape}"
        )
        assert len(probabilities.shape) == 1, (
            f"Probabilities must be a 1-D array, got {probabilities.shape}"
        )
        assert np.all(probabilities >= 0), (
            f"Probabilities must be non-negative, got {probabilities}"
        )
        assert np.all(probabilities <= 1), (
            f"Probabilities must be less than or equal to 1, got {probabilities}"
        )
        assert np.isclose(np.sum(probabilities), 1), (
            f"Probabilities must sum to 1, got {np.sum(probabilities)}"
        )
        assert len(np.unique(values)) == len(values), (
            f"Values must be unique, got {values}"
        )

        self._pmf = probabilities
        self._cmf = np.cumsum(self._pmf)
        self.values = values

    @property
    def min(self) -> int:
        return np.min(self.values).item()

    @property
    def max(self) -> int:
        return np.max(self.values).item()

    @property
    def mean(self) -> float:
        return np.dot(self.values, self._pmf)

    @property
    def variance(self) -> float:
        return np.dot(self.values**2, self._pmf) - self.mean**2

    def shift(self, offset: int) -> "DiscreteProbabilityDistribution":
        return DiscreteProbabilityDistribution(self._pmf, self.values + offset)

    def pmf(self, x: int) -> float:
        if x < self.min:
            return 0.0
        if x > self.max:
            return 0.0
        return self._pmf[np.argwhere(self.values == x).item()].item()

    def cmf(self, x: int) -> float:
        """P(X <= x)"""
        if x < self.min:
            return 0.0
        if x > self.max:
            return 1.0
        return self._cmf[np.argwhere(self.values == x).item()].item()

    def cmf_above(self, x: int) -> float:
        """P(X > x)"""
        if x < self.min:
            return 1.0
        if x > self.max:
            return 0.0
        return 1 - self.cmf(x)


def iid_linear_combination_pmf(
    probs: np.ndarray[tuple[int], np.float32],
    n: int,
    m: int,
    offset: int = -2,
    tol: float = 1e-12,
) -> tuple[np.ndarray[tuple[int], np.float32], int]:
    """
    Probability mass function (PMF) of  S =  (X_1 + ... + X_n)  -  (X_{n+1} + ... + X_{n+m}),
    where the X_i are i.i.d. with PMF `probs`
    and   value = index + offset.  (Set offset = −2 for “index − 2”.)

    Returns
    -------
    pmf   : 1-D NumPy array      – probabilities of S in ascending order
    offset_out : int            – value associated with pmf[0]

    Complexity
    ----------
    FFT-based:  O((log n + log m) · L log L)  with tiny constants.
    """

    # -------- sanity checks --------
    p = np.asarray(probs, dtype=float)
    assert p.ndim == 1 and p.size > 0, "`probs` must be a non-empty 1-D sequence."
    assert np.all(p >= 0), "Probabilities must be non-negative."
    assert n >= 0 and m >= 0 and int(n) == n and int(m) == m, (
        "`n` and `m` must be non-negative integers."
    )

    # normalise once
    p = p / p.sum()

    # --- small helpers reused from previous answer -----------------
    def _next_pow_two(x: int) -> int:
        return 1 << (x - 1).bit_length()

    def _fft_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Linear convolution via real FFT (padding to next power of two)."""
        size = _next_pow_two(len(a) + len(b) - 1)
        return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[
            : len(a) + len(b) - 1
        ]

    def _power_convolve(base: np.ndarray, k: int) -> np.ndarray:
        """k-fold convolution by binary exponentiation (delta at 0 for k=0)."""
        if k == 0:
            return np.array([1.0])
        result = np.array([1.0])
        while k:
            if k & 1:
                result = _fft_conv(result, base)
            k >>= 1
            if k:
                base = _fft_conv(base, base)
        return result

    # --- n-fold sum of  +X  ---------------------------------------
    pmf_pos = _power_convolve(p, n)
    offset_pos = n * offset

    # --- m-fold sum of  –X  ---------------------------------------
    #   -X has PMF reversed and a new offset  -(offset + (L-1))
    if m > 0:
        p_neg = p[::-1]  # reverse
        offset_neg_single = -(offset + (len(p) - 1))
        pmf_neg = _power_convolve(p_neg, m)
        offset_neg = m * offset_neg_single
    else:  # m = 0 → nothing to subtract
        pmf_neg = np.array([1.0])
        offset_neg = 0

    # --- final convolution  (+ part) * (+ (– part)) ---------------
    pmf_out = _fft_conv(pmf_pos, pmf_neg)
    offset_out = offset_pos + offset_neg

    # clean-up numerical noise
    pmf_out[pmf_out < tol] = 0.0
    pmf_out /= pmf_out.sum()

    return DiscreteProbabilityDistribution(
        pmf_out, np.arange(offset_out, offset_out + len(pmf_out))
    )


# MARK: Skyjo Game


def end_round_outcome_probabilities(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int], np.float32]:
    """Computes the probability of each player ending the round with the lowest score.

    This is from the perspective of the current player (not fixed).
    We compute this by finding the discrete probability distribution of
    each players points after reveals by as a sum iid random discrete random
    variables according to the counts reamining in the deck. We
    then compute the probability as the probability a player ends with that many
    points (PMF), and the probability of all other players having a higher score
    (1 - CMF).

    Note: this does not account for clearing probability and outcomes and also
    treats each flipped card as identical and independent rather than a single
    combination of reveals.
    """

    players = sj.get_player_count(skyjo)
    player_facedown_counts = np.array(
        [sj.get_facedown_count(skyjo, player) for player in range(players)],
        dtype=np.int16,
    )
    player_visible_scores = np.array(
        [sj.get_score(skyjo, player) for player in range(players)],
        dtype=np.int16,
    )
    deck = sj.get_deck(skyjo)
    card_probabilities = deck / np.sum(deck)
    player_distributions = {
        player: iid_linear_combination_pmf(
            card_probabilities, player_facedown_counts[player], 0
        ).shift(player_visible_scores[player])
        for player in range(players)
    }
    win_probabilities = np.zeros((players,))
    for player in range(players):
        player_distribution = player_distributions[player]
        for revealed_points in player_distribution.values:
            win_probability = player_distribution.pmf(revealed_points)
            for other_player in range(players):
                if player == other_player:
                    continue
                win_probability *= player_distributions[other_player].cmf_above(
                    revealed_points
                )
            win_probabilities[player] += win_probability
    assert np.isclose(np.sum(win_probabilities), 1), (
        f"Win probabilities should sum close to 1, got {np.sum(win_probabilities)}"
    )
    return win_probabilities
