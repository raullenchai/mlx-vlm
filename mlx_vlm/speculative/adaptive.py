"""Small request-local admission gate for a second native-MTP draft token."""

from __future__ import annotations

from collections import deque


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
