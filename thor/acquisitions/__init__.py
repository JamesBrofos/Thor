from .improvement_probability import ImprovementProbability
from .expected_improvement import ExpectedImprovement
from .upper_confidence_bound import UpperConfidenceBound
from .pure_exploration import PureExploration
from .hedge import HedgeAcquisition

acq_dict = {
    "expected_improvement": ExpectedImprovement,
    "improvement_probability": ImprovementProbability,
    "upper_confidence_bound": UpperConfidenceBound,
    "pure_exploration": PureExploration,
    "hedge": HedgeAcquisition
}
