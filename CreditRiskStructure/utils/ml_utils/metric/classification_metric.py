from CreditRiskStructure.entity.artifact_entity import ModelEvaluationArtifact
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import sys

def get_classification_score(y_true, y_pred) -> ModelEvaluationArtifact:
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        gini = 2 * roc_auc - 1

        return ModelEvaluationArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall,
            roc_auc=roc_auc,
            gini = gini
        )
    except Exception as e:
        raise CreditRiskStructureException(e, sys)