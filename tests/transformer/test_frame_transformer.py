import pandas as pd

from sklearn.datasets import load_boston
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sk_pymc3.transformers.frame_transformer import FrameTransformer

def _load_boston():
    bundle = load_boston()
    df = pd.DataFrame(
        data=bundle["data"], columns=bundle["feature_names"]
    )
    return df

def test_base_frame_transformer():
    
    df = _load_boston()

    frame_transformer = FrameTransformer(
        [
            ("scaler", StandardScaler(), ["CRIM", "INDUS"]),
            ("encoder", LabelEncoder(), ["TAX"]),
            ("pass", "passthrough", ["NOX"]),
        ]
    )
    frame_transformer.fit(df)
    df_test = frame_transformer.transform(df)

    df_new = df.copy()
    df_new["INDUS"] = StandardScaler().fit_transform(
        df_new["INDUS"].values.reshape(-1, 1)
    )
    df_new["CRIM"] = StandardScaler().fit_transform(
        df_new["CRIM"].values.reshape(-1, 1)
    )
    df_new["TAX"] = LabelEncoder().fit_transform(
        df_new["TAX"].values.reshape(-1, 1)
    )
    for col in ["INDUS", "CRIM", "TAX", "NOX"]:
        assert (
            df_new[col] == df_test[col]
        ).all()