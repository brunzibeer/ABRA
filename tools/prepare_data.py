"""Build the ABRA annotation splits from the original dataset annotations.

TODO(berna): bring the original split-generation script from the home PC and port it here.
It must produce, for each benchmark (e.g. Cityscapes -> Foggy Cityscapes):
  - domain (Objectification) training files: ``annotations_train_domain.jsonl`` (top-k classes)
  - per-class expert splits: ``experts/<cls>/annotations_train.jsonl`` and ``annotations_val.json``
  - label maps: ``label_maps/top_3.json`` and ``label_maps/experts/<cls>.json``
Acceptance check: regenerated files must be identical to the ones shipped in the GitHub release
(and to ``annotations/`` used for the Phase-0 golden runs).
Planned options from the reviews: exclude images containing unavailable classes from domain-expert
training, and a standard all-images evaluation split (see REFACTOR_PLAN.md, Phase 6).
"""


def main():
    raise NotImplementedError(
        "TODO: split generation script not ported yet (see the docstring of tools/prepare_data.py)."
    )


if __name__ == "__main__":
    main()
