"""
Expressions that are useful for evaluating performance.
"""


def all_pr(pos_scores, neg_scores):
    """
    Computes all possible precision and recall points given a list of
    scores assigned to the positive class and a list of scores assigned
    to the negative class.

    Parameters
    ----------
    pos_scores : list
        List of floats. Higher values = more likely to belong to positive
        class.
    neg_scores : list
        List of floats. Higher values = more likely to belong to negative
        class.

    Returns
    -------
    precision : list
        List of all possible precision values obtainable by varying the
        threshold of the detector.
    recall : list
        List of all possible recall values obtainable by varing the
        threshold of the detector. recall[i] is formed using the same
        threshold as precision[i]
    """

    # Attach labels to scores
    labeled_neg = [(x, 0) for x in neg_scores]
    labeled_pos = [(x, 1) for x in pos_scores]
    labeled = labeled_pos + labeled_neg
    # Sort labeled scores by descending score
    sorted_labeled = sorted(labeled, key=lambda x: -x[0])
    label_chunks = []

    # Group sorted labels into chunks of equal score
    prev_score = None
    cur_chunk = []
    for score, label in sorted_labeled:
        if prev_score is None or prev_score == score:
            cur_chunk.append(label)
        else:
            label_chunks.append(cur_chunk)
            cur_chunk = [label]
        prev_score = score
    label_chunks.append(cur_chunk)

    # Initialize with threshold set to label all inputs as negative
    precision = [1.]
    recall = [0.]
    tp = 0
    fp = 0
    fn = len(pos_scores)
    count = fn

    # Incrementally raise threshold, with each increment chosen to
    # label everything with score >= t as positive, where t is the
    # score of the next tied chunk
    for label_chunk in label_chunks:
        for label in label_chunk:
            if label:
                fn -= 1
                tp += 1
            else:
                fp += 1
        ap = tp + fp
        if ap == 0:
            precision.append(1.)
        else:
            precision.append(float(tp)/float(ap))
        recall.append(float(tp)/count)

    return precision, recall
