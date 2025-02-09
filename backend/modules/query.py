def query_settings(lang: str, algofam: str, platform: str, tuning_limit: int) -> dict:
    """
    Generate preferred training settings for new dataset.

    :param lang: Preferred language.
    :param algofam: Must be in 3-char format.
    :param platform: Must match one of the predefined platforms or option "other".
    :param tuning_limit: Threshold number of hyperparameters that are considered acceptable.
    :return: Dictionary of technical settings for training the ML model.
    """
    if algofam not in ["DLN", "RFR", "DTR", "NBY", "LGR", "SVM", "KNN", "GBE", "GLM"]:
        raise ValueError("Error: Algorithm unknown.")

    pform = ""
    if platform.lower() in ["scikit", "sklearn", "scikit-learn"]:
        pform = "scikit"
    elif platform.lower() in ["h2o", "h2o_cluster_version", "mojo"]:
        pform = "h2o"
    elif platform.lower() in ["rweka", "weka", "r"]:
        pform = "weka"

    return {"language": lang, "algorithm": algofam, "platform": pform, "hparams": tuning_limit}
