"""Contains the implementation for different metrics.

Code based on VisSeq - https://github.com/facebookresearch/vizseq
@inproceedings{wang2019vizseq,
  title = {VizSeq: A Visual Analysis Toolkit for Text Generation Tasks},
  author = {Changhan Wang, Anirudh Jain, Danlu Chen, Jiatao Gu},
  booktitle = {In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year = {2019},
}
"""
from typing import Dict, List, Optional, Tuple

from vizseq.scorers.bleu import BLEUScorer
from vizseq.scorers.cider import CIDErScorer
from vizseq.scorers.meteor import METEORScorer
from vizseq.scorers.rouge import Rouge1Scorer, Rouge2Scorer
from vizseq.scorers.wer import WERScorer


class Scorers():
    """Setup evaluation scorers and compute the metrics."""
    def __init__(self, scorers: List[str], num_workers: int = 0) -> None:
        """
        Args:
            scorers (List[str]): A list of metrics to compute.
                Possible options are `bleu`, `wer`, `rogue1`, `rogue2`, `meteor`, `cider`
            num_workers (int): The amount of subprocesses for computing the scores.
        """
        self.scorers = {}
        for scorer in scorers:
            scorer = scorer.lower()
            if scorer == 'bleu':
                self.scorers[scorer] = BLEUScorer(n_workers=num_workers)
            if scorer == 'wer':
                self.scorers[scorer] = WERScorer(n_workers=num_workers)
            if scorer == 'rogue1':
                self.scorers[scorer] = Rouge1Scorer(n_workers=num_workers)
            if scorer == 'rogue2':
                self.scorers[scorer] = Rouge2Scorer(n_workers=num_workers)
            if scorer == 'meteor':
                self.scorers[scorer] = METEORScorer(n_workers=num_workers)
            if scorer == 'cider':
                self.scorers[scorer] = CIDErScorer(n_workers=num_workers)

    def compute_scores(self, hypothesis: List[str], references: List[List[str]],
                       tags: Optional[List[List[str]]] = None) -> Dict[str, Tuple[Optional[float], Optional[List[float]], Optional[Dict[str, float]]]]:
        """Compute scores from all listed scorers.

        Args:
            hypothesis (List[str]): The list of sentences predicted by the model.
            references (List[List[str]]): The list of list of references with dimensions (reference_sets, sentences).
                Each reference set needs to have the same amout of sentences like the hypothesis!
            tags (Optional[List[List[str]]]): If you want to tag the references and compute a combined score of taged hypothesis and references.

        Returns:
            Dict[str, Tuple[Optional[float], Optional[List[float]]], Optional[Dict[str, float]]]:
                str: The name of the scorer.
                tuple:
                    - corpus_score: Optional[float] = None
                    - sent_scores: Optional[List[float]] = None
                    - group_scores: Optional[Dict[str, float]] = None
        """
        scores = {}
        # compute all scores
        for name, scorer in self.scorers.items():
            scores[name] = scorer.score(hypothesis, references)

        return scores
