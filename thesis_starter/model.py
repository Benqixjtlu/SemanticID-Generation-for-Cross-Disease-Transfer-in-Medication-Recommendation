"""Learn next-token counts from synthetic text and use the catalogue grammar."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import log

from .data import Visit
from .decoding import CatalogDecoder, DecodeState, Token

# Count how many complete drugs precede the current drug prefix.
Context = tuple[int, tuple[int, ...]]


@dataclass(frozen=True)
class SearchPath:
    state: DecodeState
    log_probability: float = 0.0


class PrefixCountModel:
    """A fitted next-token model for the space-separated cues in the fixture."""

    def __init__(self, decoder: CatalogDecoder) -> None:
        self.decoder = decoder
        self.context_counts: dict[Context, Counter[Token]] = {}
        self.cue_counts: dict[tuple[str, Context], Counter[Token]] = {}

    def learn(self, visits: Iterable[Visit]) -> None:
        """Advance with the gold token and count each observed transition."""
        examples = []
        for visit in visits:
            if visit.discharge_medications is not None:
                # Validate all labelled rows before updating either count table.
                tokens = self.decoder.serialize(visit.discharge_medications)
                examples.append((set(visit.text.split()), tokens))

        for cues, tokens in examples:
            state = DecodeState()
            for token in tokens:
                context = (len(state.drugs), state.prefix)
                self.context_counts.setdefault(context, Counter())[token] += 1
                for cue in cues:
                    self.cue_counts.setdefault((cue, context), Counter())[token] += 1
                state = self.decoder.advance(state, token)

    def token_probabilities(self, text: str, state: DecodeState) -> dict[Token, float]:
        """Mask to legal tokens, smooth their counts, then divide by the total."""
        allowed = self.decoder.allowed(state)
        if not allowed:
            return {}

        context = (len(state.drugs), state.prefix)
        counts: Counter[Token] = Counter()
        for cue in set(text.split()):
            known = self.cue_counts.get((cue, context))
            if known is not None:
                counts.update(known)
        if all(counts[token] == 0 for token in allowed):
            counts = self.context_counts.get(context, Counter())

        weights = {token: counts[token] + 1 for token in allowed}
        total = sum(weights.values())
        return {token: weight / total for token, weight in weights.items()}

    def predict(self, text: str, *, beam_width: int = 3) -> tuple[str, ...]:
        """Follow the model's own paths and keep finished results separately."""
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        active = [SearchPath(DecodeState())]
        best_finished: SearchPath | None = None

        # Each path terminates after at most one traversal per allowed drug.
        while active:
            extensions = []
            for path in active:
                for token, probability in self.token_probabilities(text, path.state).items():
                    step = SearchPath(
                        self.decoder.advance(path.state, token),
                        path.log_probability + log(probability),
                    )
                    if step.state.finished:
                        if best_finished is None or step.log_probability > best_finished.log_probability:
                            best_finished = step
                    else:
                        extensions.append(step)
            active = sorted(extensions, key=lambda path: -path.log_probability)[:beam_width]

        if best_finished is None:
            raise RuntimeError("the catalogue grammar did not produce a finished path")
        return best_finished.state.drugs
