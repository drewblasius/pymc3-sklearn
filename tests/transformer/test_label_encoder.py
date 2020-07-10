from sk_pymc3.transformers.label_encoder import MissingLabelEncoder


def test_missing_label_encoder():

    x = ["a", "b", "b"]

    mle = MissingLabelEncoder().fit(
        ["a", "b", "b"]
    )

    cases = [
        (["a", "b", "c"], [1, 2, 0]),
        (["b", "c"], [2, 0]),
        (["b", "d"], [2, 0]),
    ]

    for x, y in cases:

        yh = mle.transform(x)
        print(yh)
        assert list(yh) == y