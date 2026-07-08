# Armut Association Rule-Based Recommender System

This project develops an association rule-based recommendation system for Armut, an online service marketplace that connects customers with service providers.

The objective is to analyze the services purchased together by users and recommend relevant services using Association Rule Learning.

## Business Problem

Armut provides access to services such as cleaning, transportation, renovation, private lessons, and technical support.

Using historical customer-service interaction data, this project aims to answer the following question:

> Which services can be recommended to a customer based on a service they previously purchased?

Since the dataset does not contain a traditional invoice or basket identifier, each customer's monthly service purchases are considered a separate basket.

## Dataset

The dataset contains **162,523 service transactions** and four original variables.

| Variable | Description |
|---|---|
| `UserId` | Unique customer identifier |
| `ServiceId` | Anonymized service identifier |
| `CategoryId` | Anonymized service category identifier |
| `CreateDate` | Date and time when the service was purchased |

A `ServiceId` may represent different services under different categories. Therefore, `ServiceId` and `CategoryId` are combined to create a unique service code.

## Project Workflow

### 1. Data Exploration

The dataset was examined in terms of:

- Dataset dimensions
- Variable types
- Missing values
- Number of unique users, services, and categories
- Descriptive statistics

The dataset does not contain any missing values.

### 2. Creating Unique Service Codes

`ServiceId` and `CategoryId` were combined to create the `Product_SC` variable.

```python
df["Product_SC"] = (
    "SC_"
    + df["ServiceId"].astype(str)
    + "_"
    + df["CategoryId"].astype(str)
)
```

### 3. Creating Monthly Customer Baskets

Because the dataset does not contain an invoice identifier, a basket was defined as all services purchased by a customer within the same month.

First, a year-month variable was created:

```python
df["date"] = pd.to_datetime(
    df["CreateDate"]
).dt.to_period("M")
```

Then, `UserId` and the monthly date variable were combined:

```python
df["Invoice_Id"] = (
    df["UserId"].astype(str)
    + "_"
    + df["date"].astype(str)
)
```

For example:

```text
UserId: 25446
Month: 2017-08
Basket ID: 25446_2017-08
```

### 4. Creating the Basket-Service Matrix

A binary basket-service matrix was created.

- Rows represent monthly customer baskets.
- Columns represent service codes.
- `1` indicates that the service exists in the basket.
- `0` indicates that the service does not exist in the basket.

### 5. Frequent Itemset Generation

The Apriori algorithm was applied using a minimum support threshold of `0.01`.

```python
frequent_itemsets_SC = apriori(
    product_in_cart_df,
    min_support=0.01,
    use_colnames=True
)
```

A total of **56 frequent itemsets** were identified.

### 6. Association Rule Generation

Association rules were generated from the frequent itemsets.

```python
rules_SC = association_rules(
    frequent_itemsets_SC,
    metric="support",
    min_threshold=0.01
)
```

The rules were evaluated using the following metrics:

| Metric | Description |
|---|---|
| Support | Frequency of the service combination in all baskets |
| Confidence | Probability of purchasing the consequent service after purchasing the antecedent |
| Lift | Strength of the relationship compared with random occurrence |
| Leverage | Difference between observed and expected co-occurrence |

Rules with a lift value greater than `1` represent positive service relationships.

## Dataset Summary

| Metric | Value |
|---|---:|
| Number of transactions | 162,523 |
| Number of unique users | 24,826 |
| Number of unique service-category codes | 50 |
| Number of monthly baskets | 71,220 |
| Number of frequent itemsets | 56 |
| Minimum support threshold | 0.01 |

## Important Association Rules

Some of the strongest association rules obtained from the analysis are:

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| `SC_22_0` | `SC_25_0` | 0.011120 | 0.234043 | 5.456141 |
| `SC_25_0` | `SC_22_0` | 0.011120 | 0.259247 | 5.456141 |
| `SC_38_4` | `SC_9_4` | 0.010067 | 0.151234 | 3.653623 |
| `SC_9_4` | `SC_38_4` | 0.010067 | 0.243216 | 3.653623 |
| `SC_15_1` | `SC_33_4` | 0.011233 | 0.092861 | 3.400299 |

For example, the lift value of `5.456` between `SC_22_0` and `SC_25_0` shows that these services are purchased together much more frequently than expected by chance.

## Recommendation Function

A reusable recommendation function was developed to generate service recommendations from the association rules.

```python
def arl_recommender(
    rules_df,
    product_id,
    rec_count=4,
    min_lift=1.0,
    min_confidence=0.0
):
    product_rules = rules_df[
        rules_df["antecedents"].apply(
            lambda antecedent: product_id in antecedent
        )
    ].copy()

    product_rules = product_rules[
        (product_rules["lift"] > min_lift)
        & (product_rules["confidence"] >= min_confidence)
    ]

    product_rules = product_rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    )

    recommendations = []
    seen = set()

    for consequent_set in product_rules["consequents"]:
        for recommended_product in consequent_set:

            if (
                recommended_product != product_id
                and recommended_product not in seen
            ):
                seen.add(recommended_product)
                recommendations.append(recommended_product)

            if len(recommendations) >= rec_count:
                return recommendations

    return recommendations
```

The function:

1. Finds rules in which the selected service appears in the antecedent.
2. Filters rules according to lift and confidence thresholds.
3. Sorts the rules by lift, confidence, and support.
4. Removes duplicate recommendations.
5. Prevents the selected service from being recommended to itself.

## Example Recommendation

Recommendations for a customer who previously purchased the `SC_2_0` service:

```python
recommendations = arl_recommender(
    rules_df=rules_SC,
    product_id="SC_2_0",
    rec_count=4
)

print(
    "SC_2_0 hizmeti için öneriler:",
    recommendations
)
```

Output:

```text
SC_2_0 hizmeti için öneriler:
['SC_22_0', 'SC_25_0', 'SC_15_1', 'SC_13_11']
```

According to the generated association rules, the services most relevant to `SC_2_0` are:

1. `SC_22_0`
2. `SC_25_0`
3. `SC_15_1`
4. `SC_13_11`

## Technologies Used

- Python
- Pandas
- Matplotlib
- MLxtend
- Apriori Algorithm
- Association Rule Learning
- Jupyter Notebook

## Installation

Install the necessary libraries using:

```bash
pip install pandas matplotlib mlxtend jupyter
```

## Dataset Availability

The dataset used in this project is not included in this GitHub repository.

## Project Structure

```text
Armut-ARL-Recommender-System/
│
├── armut_arl.ipynb
└── README.md
```

## Limitations

- Service and category names are anonymized.
- Recommendations are based only on historical co-occurrence patterns.
- Customer preferences, ratings, and demographic information are not available.
- Monthly purchases are assumed to represent a single basket.
- Services without sufficient support may not receive recommendations.

## Conclusion

This project demonstrates how Association Rule Learning can be used to build a service recommendation system.

By transforming monthly customer purchases into baskets, frequent service combinations were identified with the Apriori algorithm. The resulting association rules were then used to recommend complementary services based on lift, confidence, and support values.

The developed recommendation function can support cross-selling activities and help customers discover services that are frequently purchased together.

# Ezgi Dönmez
