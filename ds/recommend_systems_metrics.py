from typing import Optional
import numpy as np


def cls_metrics(y_true: list, y_pred: list, top_k: Optional[int] = None) -> dict[str, float]:
    """Расчет классификационных метрик для первых топ-k предсказаний.

    Args:
        y_true: истинные классы.
        y_pred: предсказания, отсортированные в порядке важные (самые важные вначале),
            предсказания должны содержать как минимум top_k элементов.
        top_k: кол-во первых предсказаний, по которым будут считаться метрики.
            Если top_k = None, то берется срез размером с y_true.

    Returns:
        dict: словарь с расчитанными метриками.

    Examples:
        >>> cls_metrics([1, 2, 3, 4], [1, 4, 5, 6, 7], 2)  # doctest: +ELLIPSIS
        {'precision': 0.5, 'recall': 1.0, 'f1': 0.666...}

        >>> cls_metrics([1, 2, 3, 4], [1, 4, 5, 6, 7], 4)
        {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}

    """
    top_k = top_k or len(y_true)
    if len(y_pred) < top_k:
        raise ValueError(f'y_pred must contain at least top_k={top_k} elements.')

    y_pred = y_pred[:top_k]

    results = {
        'precision': precision(y_true, y_pred),
        'recall': recall(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

    return results


def avg_precision(y_true: list, y_pred: list, max_k: int = 20) -> float:
    """Расчет усредненной точности.
    Расчитывается среднее между разными precision,
    каждое из которых берется по топ i пердсказний,
    где i берется от 1 до max_k.

    Args:
        y_true: истинные классы.
        y_pred: предсказания, отсортированные в порядке важные (самые важные вначале).
        max_k: максимальное кол-во первых предсказаний, по которым будут считаться метрики.

    Returns:
        float: значение метрики.

    Examples:
        >>> avg_precision([1, 2, 3], [1, 4, 5])  # doctest: +ELLIPSIS
        0.333...

        >>> avg_precision([1, 2, 3], [4, 5, 1])  # doctest: +ELLIPSIS
        0.111...

        >>> avg_precision([1, 2, 3], [5, 1, 3])  # doctest: +ELLIPSIS
        0.388...

        >>> avg_precision([1, 2, 3], [2, 1, 3])
        1.0

    """
    y_pred = y_pred[:max_k]
    m = min(len(y_true), len(y_pred))
    precision = 0
    relevant_preds_num = 0
    for i, pred in enumerate(y_pred, start=1):
        if pred in y_true:
            relevant_preds_num += (pred in y_true)
            precision += relevant_preds_num / i

    return precision / m


def precision(y_true: list, y_pred: list) -> float:
    """Возвращает точность рекомендательной системы.

    Args:
        y_true: истинные классы.
        y_pred: предсказания модели.

    Examples:
        >>> precision([1, 2, 3], [1, 4, 5])  # doctest: +ELLIPSIS
        0.333...

        >>> precision([1, 2, 3], [1, 2, 5, 4])  # doctest: +ELLIPSIS
        0.666...

    """
    return len(np.intersect1d(y_true, y_pred)) / len(y_true)


def recall(y_true: list, y_pred: list) -> float:
    """Возвращает точность рекомендательной системы.

    Args:
        y_true: истинные классы.
        y_pred: предсказания модели.

    Examples:
        >>> recall([1, 2, 3], [1, 4, 5])  # doctest: +ELLIPSIS
        0.333...

        >>> recall([1, 2, 3], [1, 2, 5, 4])
        0.5

    """
    return len(np.intersect1d(y_true, y_pred)) / len(y_pred)


def f1_score(y_true: list, y_pred: list) -> float:
    """Возвращает F1-score рекомендательной системы.

    Args:
        y_true: истинные классы.
        y_pred: предсказания модели.

    Examples:
        >>> f1_score([1, 2, 3], [1, 4, 5])  # doctest: +ELLIPSIS
        0.333...

        >>> f1_score([1, 2, 3], [1, 2, 5, 4]) # doctest: +ELLIPSIS
        0.5714...
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
