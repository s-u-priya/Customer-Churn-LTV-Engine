def calculate_ltv(
    monthly,
    tenure,
    churn_probability
):

    retention = (
        1 - churn_probability
    )

    ltv = (
        monthly *
        max(tenure, 1) *
        retention
    )

    return round(
        ltv,
        2
    )