"""Serialize and decode catalogued drug sets with shared prefix rules."""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


STOP = "<STOP>"
Token = int | str


@dataclass(frozen=True)
class DecodeState:
    drugs: tuple[str, ...] = ()
    prefix: tuple[int, ...] = ()
    finished: bool = False


class CatalogDecoder:
    """Only allow complete drug paths in a fixed, strictly increasing rank."""

    def __init__(
        self,
        paths: Mapping[str, tuple[int, ...]],
        ranking: Sequence[str] | None = None,
    ) -> None:
        if not paths or any(not isinstance(drug, str) or not drug.strip() for drug in paths):
            raise ValueError("The catalogue requires at least one nonempty drug name.")
        if any(
            not isinstance(path, tuple) or not path
            or any(type(code) is not int for code in path)
            for path in paths.values()
        ):
            raise ValueError("Drug paths must be nonempty integer tuples.")
        if len({len(path) for path in paths.values()}) != 1:
            raise ValueError("All drug paths must have the same length.")
        if len(set(paths.values())) != len(paths):
            raise ValueError("Complete drug paths must be unique.")

        self.ranking = tuple(sorted(paths) if ranking is None else ranking)
        if len(self.ranking) != len(paths) or set(self.ranking) != set(paths):
            raise ValueError("The ranking must include each drug exactly once.")
        self.paths = MappingProxyType(dict(paths))
        self._rank = {drug: rank for rank, drug in enumerate(self.ranking)}
        self._drug_for_path = {path: drug for drug, path in self.paths.items()}

        # Each prefix indexes its outgoing codes. On each branch, store the
        # largest reachable drug rank so we can skip branches with no eligible drug.
        self._branches: dict[tuple[int, ...], dict[int, int]] = {}
        for rank, drug in enumerate(self.ranking):
            path = self.paths[drug]
            for level, code in enumerate(path):
                self._branches.setdefault(path[:level], {})[code] = rank

    def allowed(self, state: DecodeState) -> tuple[Token, ...]:
        """Return the legal next codes, followed by STOP at drug boundaries."""
        last_rank = -1
        for drug in state.drugs:
            rank = self._rank.get(drug, -1)
            if rank <= last_rank:
                raise ValueError("Completed drugs must follow the catalogue ranking.")
            last_rank = rank

        if state.finished:
            if state.prefix:
                raise ValueError("A finished state cannot have an unfinished drug path.")
            return ()

        branches = self._branches.get(state.prefix)
        if branches is None:
            raise ValueError("The state has an invalid unfinished drug path.")
        codes = tuple(sorted(
            code for code, max_rank in branches.items() if max_rank > last_rank
        ))
        if not state.prefix:
            return (*codes, STOP)
        if not codes:
            raise ValueError("No ranked drug can complete this prefix.")
        return codes

    def advance(self, state: DecodeState, token: Token) -> DecodeState:
        """Take one legal step and collect a drug when its path is complete."""
        if (token != STOP and type(token) is not int) or token not in self.allowed(state):
            raise ValueError(f"Illegal token {token!r} for this decoding state.")
        if not isinstance(token, int):
            return DecodeState(drugs=state.drugs, finished=True)

        prefix = (*state.prefix, token)
        completed_drug = self._drug_for_path.get(prefix)
        if completed_drug is not None:
            return DecodeState(drugs=(*state.drugs, completed_drug))
        return DecodeState(drugs=state.drugs, prefix=prefix)

    def serialize(self, drugs: Collection[str]) -> tuple[Token, ...]:
        """Write one canonical token sequence for the supplied drug set."""
        if isinstance(drugs, str):
            raise ValueError("Pass a collection of drug names, such as {'A'}.")
        drug_set = set(drugs)
        unknown = drug_set.difference(self.paths)
        if unknown:
            raise ValueError(f"Drugs outside the catalogue: {sorted(unknown)}")
        tokens: list[Token] = []
        for drug in self.ranking:
            if drug in drug_set:
                tokens.extend(self.paths[drug])
        tokens.append(STOP)
        return tuple(tokens)

    def decode(self, tokens: Iterable[Token]) -> tuple[str, ...]:
        """Parse a full sequence, applying the same rules as generation."""
        state = DecodeState()
        for token in tokens:
            state = self.advance(state, token)
        if not state.finished:
            raise ValueError("The sequence must end with STOP.")
        return state.drugs
