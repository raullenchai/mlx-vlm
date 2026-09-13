"""Small request-local admission gate for a second native-MTP draft token."""

from __future__ import annotations

from collections import deque


class _CostEwma:
    """Small, stall-resistant cost estimate for one draft depth."""

    def __init__(self, *, alpha: float = 0.3, clamp_fraction: float = 0.25):
        self.alpha = alpha
        self.clamp_fraction = clamp_fraction
        self.value = 0.0
        self.samples = 0

    def observe(self, wall_ms: float) -> None:
        if wall_ms <= 0.0:
            return
        if not self.samples:
            self.value = wall_ms
        else:
            limit = self.clamp_fraction * self.value
            innovation = max(-limit, min(limit, wall_ms - self.value))
            self.value += self.alpha * innovation
        self.samples += 1


class K23EvController:
    """Pick one or two MTP drafts from measured tokens-per-compute-time.

    Unlike :class:`K23AcceptanceGate`, this controller has no model-specific
    acceptance cutoff.  It learns the compute cost of each width and the
    conditional acceptance probability of each draft position, then compares
    ``expected committed tokens / measured milliseconds``.
    """

    def __init__(
        self,
        *,
        seed_samples: int = 4,
        acceptance_alpha: float = 0.1,
        probe_interval: int = 64,
    ):
        if seed_samples < 1:
            raise ValueError("seed_samples must be positive")
        if not 0.0 < acceptance_alpha <= 1.0:
            raise ValueError("acceptance_alpha must be in (0, 1]")
        if probe_interval < 1:
            raise ValueError("probe_interval must be positive")
        self.seed_samples = seed_samples
        self.acceptance_alpha = acceptance_alpha
        self.probe_interval = probe_interval
        self.cost = {1: _CostEwma(), 2: _CostEwma()}
        self.rate = {1: 1.0, 2: 1.0}
        self.seen = {1: 0, 2: 0}
        self.rounds = 0
        self._since_probe = 0
        self.depth = 1

    def _selected(self) -> int:
        if any(self.cost[depth].samples < self.seed_samples for depth in (1, 2)):
            return 1
        p1 = self.rate[1]
        expected = {
            1: 1.0 + p1,
            2: 1.0 + p1 + p1 * self.rate[2],
        }
        return max((1, 2), key=lambda depth: expected[depth] / self.cost[depth].value)

    def pick(self) -> int:
        for depth in (1, 2):
            if self.cost[depth].samples < self.seed_samples:
                self.depth = depth
                return depth
        selected = self._selected()
        self._since_probe += 1
        if self._since_probe >= self.probe_interval:
            self._since_probe = 0
            self.depth = 3 - selected
        else:
            self.depth = selected
        return self.depth

    def record(
        self, *, depth: int, accepted: int, drafted: int, wall_ms: float
    ) -> None:
        if depth not in (1, 2) or drafted != depth:
            return
        if accepted < 0 or accepted > drafted:
            raise ValueError("accepted drafts must be within the drafted width")
        self.rounds += 1
        self.cost[depth].observe(wall_ms)
        for position in range(1, depth + 1):
            if accepted < position - 1:
                break
            outcome = 1.0 if accepted >= position else 0.0
            if not self.seen[position]:
                self.rate[position] = outcome
            else:
                self.rate[position] += self.acceptance_alpha * (
                    outcome - self.rate[position]
                )
            self.seen[position] += 1

    def summary(self) -> str:
        return (
            f"rounds={self.rounds} selected_depth={self._selected()} "
            f"cost_ms=1:{self.cost[1].value:.2f},2:{self.cost[2].value:.2f} "
            f"conditional_acceptance=1:{self.rate[1]:.3f},2:{self.rate[2]:.3f}"
        )


class K23AcceptanceGate:
    """Keep two drafts only when an initial request-local sample supports it.

    The first draft token is already qualified as profitable for the target.
    This gate answers only whether adding the second draft token is worthwhile.
    It deliberately samples the actual two-draft path: first-position
    acceptance alone cannot predict the conditional acceptance of position 2.
    """

    def __init__(
        self,
        *,
        sample_rounds: int = 12,
        min_acceptance: float = 0.65,
        window_rounds: int = 64,
    ):
        if sample_rounds < 1:
            raise ValueError("sample_rounds must be positive")
        if not 0.0 <= min_acceptance <= 1.0:
            raise ValueError("min_acceptance must be between zero and one")
        if window_rounds < sample_rounds:
            raise ValueError("window_rounds must cover the initial sample")
        self.sample_rounds = sample_rounds
        self.min_acceptance = min_acceptance
        self.window = deque(maxlen=window_rounds)
        self.rounds = 0
        self.accepted = 0
        self.drafted = 0
        self.depth = 2

    def pick(self) -> int:
        return self.depth

    def record(self, *, depth: int, accepted: int, drafted: int) -> None:
        """Record one complete round; partial terminal widths do not classify."""
        if self.depth != 2 or depth != 2 or drafted != 2:
            return
        if accepted < 0 or accepted > drafted:
            raise ValueError("accepted drafts must be within the drafted width")
        self.rounds += 1
        self.accepted += accepted
        self.drafted += drafted
        self.window.append(accepted)
        if len(self.window) >= self.sample_rounds:
            rate = sum(self.window) / (2 * len(self.window))
            if rate < self.min_acceptance:
                self.depth = 1

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def window_acceptance(self) -> float:
        return sum(self.window) / (2 * len(self.window)) if self.window else 0.0

    def summary(self) -> str:
        return (
            f"sampled={self.rounds} acceptance={self.acceptance:.3f} "
            f"window_acceptance={self.window_acceptance:.3f} "
            f"selected_depth={self.depth}"
        )
