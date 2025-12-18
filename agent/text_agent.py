from agent.hypothesis_agent import hypothesis_agent

FEATURE_MAP = {
    "mileage": "AVG_ANNUAL_MILEAGE",
    "annual mileage": "AVG_ANNUAL_MILEAGE",
    # Telematics-like score present in the provided dataset
    "telematics": "RETRO_10SEC_GPS_SCORE_V1_SCORE",
    "gps score": "RETRO_10SEC_GPS_SCORE_V1_SCORE",
    "retro": "RETRO_10SEC_GPS_SCORE_V1_SCORE",
}

def run_from_text(df, text):
    text = text.lower()

    for key, feature in FEATURE_MAP.items():
        if key in text:
            return hypothesis_agent(
                df,
                feature=feature,
                expected_direction="increasing"
            )

    return {"error": "Could not identify feature from text"}
