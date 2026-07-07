import sys
import numpy as np

from CreditRiskStructure.exception.exception import CreditRiskStructureException

# Standard scorecard scaling parameters
BASE_SCORE = 600
BASE_ODDS = 50       
PDO = 20             

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def probability_to_score(probability_default, min_score: int = 300, max_score: int = 850):
    try:
        arr = np.asarray(probability_default, dtype=float)
        p = np.clip(arr, 1e-6, 1 - 1e-6)
        odds_of_good = (1 - p) / p
        score = OFFSET + FACTOR * np.log(odds_of_good)
        score = np.clip(score, min_score, max_score)
        if score.ndim == 0:
            return int(round(float(score)))
        return np.round(score).astype(int)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
