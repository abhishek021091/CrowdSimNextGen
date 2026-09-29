"""Training / evaluation pipeline for the official-port CrowdNav++ policy
(``navcore.policies.original``).

Recommended entry points::

    python -m navcore.training.original.train_original
    python -m navcore.training.original.evaluate_original

Deliberately import-light: nothing heavy (gymnasium, shapely, rvo2) is
imported here so ``navcore.training.original.checkpoint`` can be used alone.
"""
