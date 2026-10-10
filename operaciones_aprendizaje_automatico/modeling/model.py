from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from ..features import create_features


def build_model(args) -> Pipeline:
    class_weight = (
        None if args.class_weight == "none"
        else "balanced"
    )

    pipeline = Pipeline([
        ("preprocesamiento", create_features()),
        ("modelo", LogisticRegression(
                                        solver="saga",
                                        C=args.C,
                                        l1_ratio=args.l1_ratio,
                                        class_weight=class_weight,
                                        max_iter=args.max_iter,
                                        random_state=args.random_state,
                                    )
        )])

    return pipeline