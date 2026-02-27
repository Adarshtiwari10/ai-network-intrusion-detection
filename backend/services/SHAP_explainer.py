import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_explainer(model):
    return shap.TreeExplainer(model)

def generate_shap_analysis(explainer, packet_df, feature_names, prediction):

    shap_output = explainer(packet_df)

    if len(shap_output.values.shape) == 3:
        shap_vector = shap_output.values[0, :, 1]
    else:
        shap_vector = shap_output.values[0]

    # Prepare impact dataframe
    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_vector,
        "Actual Value": packet_df.iloc[0].values
    })

    impact_df["AbsImpact"] = impact_df["Impact"].abs()
    impact_df = impact_df.sort_values(by="AbsImpact", ascending=False)

    top_impacts = impact_df.head(5)

    explanation_text = build_explanation_text(top_impacts, prediction)

    return shap_vector, explanation_text


def build_explanation_text(top_impacts, prediction):

    def categorize_feature(feature_name):
        if "Packets" in feature_name or "Bytes" in feature_name:
            return "Traffic Volume"
        elif "Length" in feature_name or "Segment" in feature_name:
            return "Packet Size Characteristics"
        elif "IAT" in feature_name:
            return "Timing Behavior"
        elif "Flag" in feature_name:
            return "Protocol Flags"
        elif "Win" in feature_name:
            return "TCP Window Behavior"
        else:
            return "General Network Behavior"

    prediction_label = "ATTACK" if prediction == 1 else "BENIGN"

    grouped_reasons = {}

    for _, row in top_impacts.iterrows():
        category = categorize_feature(row["Feature"])
        grouped_reasons.setdefault(category, []).append(row)

    explanation_text = (
        f"The model classified this packet as {prediction_label} "
        f"based on the following behavioral indicators:\n\n"
    )

    for category, rows in grouped_reasons.items():
        explanation_text += f"**{category}:**\n"
        for row in rows:
            direction = "increased" if row["Impact"] > 0 else "decreased"
            explanation_text += (
                f"• {row['Feature']} = {row['Actual Value']} "
                f"{direction} attack probability "
                f"(impact score: {row['Impact']:.4f})\n"
            )
        explanation_text += "\n"

    if prediction == 1:
        explanation_text += "Overall, these patterns led to attack classification."
    else:
        explanation_text += "Overall, these patterns led to benign classification."

    return explanation_text