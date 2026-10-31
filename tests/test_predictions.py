import numpy as np

from surgclip_explorer.classifiers import PredictionResult
from surgclip_explorer.config import SURGICAL_CLASSES


def test_topk_follows_descending_scores() -> None:
    scores = np.linspace(-0.2, 0.3, 11, dtype=np.float32)
    ranking = tuple(np.argsort(-scores).tolist())
    result = PredictionResult(tuple(SURGICAL_CLASSES), scores, ranking, 1.0, 2.0, "cosine similarity")
    assert len(result.scores) == 11
    assert result.topk(3)[0][0] == SURGICAL_CLASSES[-1]
    assert [score for _, score in result.topk(3)] == sorted(
        [score for _, score in result.topk(3)], reverse=True
    )
