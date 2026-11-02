"""Cooperative evaluation: one outstanding request per independent search/game."""

from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass

from skyjo.engine import game as sj

from .evaluator import Evaluator, Prediction


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    evaluator: Evaluator
    states: Sequence[sj.Skyjo]


type EvaluationWork[T] = Generator[EvaluationRequest, list[Prediction], T]


def evaluate[T](work: EvaluationWork[T]) -> T:
    """Drive the same resumable algorithm synchronously, without batching peers."""
    try:
        request = next(work)
        while True:
            request = work.send(request.evaluator.evaluate(request.states))
    except StopIteration as completed:
        return completed.value
    finally:
        work.close()


def evaluate_concurrently[T](
    tasks: Iterable[EvaluationWork[T]], concurrency: int
) -> list[T]:
    """Batch requests by evaluator identity and refill slots as tasks finish.

    Results retain input order. Each task owns its RNG and completes a real
    backup before selecting its next leaf; there are no virtual visits or threads.
    Evaluators must be independent of request ordering for batching equivalence.
    Neural batch shapes can still introduce floating-point differences.
    """
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    pending = iter(tasks)
    live = {}
    results = []

    def refill():
        while len(live) < concurrency:
            work = next(pending, None)
            if work is None:
                break
            index = len(results)
            results.append(None)
            try:
                request = next(work)
            except StopIteration as completed:
                results[index] = completed.value
            else:
                live[index] = (work, request)

    try:
        refill()
        while live:
            groups = {}
            for index, (_, request) in live.items():
                groups.setdefault(id(request.evaluator), []).append(index)
            answers = {}
            for indices in groups.values():
                evaluator = live[indices[0]][1].evaluator
                states = [s for i in indices for s in live[i][1].states]
                predictions = evaluator.evaluate(states)
                if len(predictions) != len(states):
                    raise ValueError("Evaluator returned the wrong prediction count")
                offset = 0
                for index in indices:
                    count = len(live[index][1].states)
                    answers[index] = predictions[offset : offset + count]
                    offset += count
            for index, (work, _) in list(live.items()):
                try:
                    live[index] = (work, work.send(answers[index]))
                except StopIteration as completed:
                    results[index] = completed.value
                    del live[index]
            refill()
    finally:
        for work, _ in live.values():
            work.close()
    return results
