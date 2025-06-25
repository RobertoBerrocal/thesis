# =============================
# 1. IMPORTING LIBRARIES
# =============================

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklift.models import ClassTransformation,TwoModels
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklift.metrics import uplift_at_k, qini_auc_score,uplift_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklift.metrics import qini_curve
from sklift.metrics import uplift_curve

# ============================
# 2. UNSUPERVISED LEARNING
# ============================

rfm_df = pd.read_parquet('csv_export/RFM.parquet')

rfm_df.info()
rfm_df.dropna(inplace=True)

categorical_cols = [
    'FAVOURITE_STORE',
    'FAVOURITE_STORE_TYPE',
    'FAVOURITE_PAYMENT_METHOD',
    'FAVORITE_CATEGORY',
    'FAVORITE_SUB_CATEGORY',
    'FAVORITE_WEEKDAY'
]

numerical_cols = [
    col for col in rfm_df.select_dtypes(include=['int64', 'float64']).columns
    if col != 'CUSTOMER_ID' and col != 'FAVORITE_HOUR'  
]

categorical_cols.append('FAVORITE_HOUR')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(rfm_df)

num_features = numerical_cols
cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
all_features = numerical_cols + cat_features

X_df = pd.DataFrame(X_processed, columns=all_features, index=rfm_df.index) # type: ignore

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  
rfm_df['Anomaly_Score'] = iso_forest.fit_predict(X_df)

rfm_df['Churn_Label_IsoForest'] = (rfm_df['Anomaly_Score'] == -1).astype(int)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
rfm_df['LOF_Score'] = lof.fit_predict(X_df)

rfm_df['Churn_Label_LOF'] = (rfm_df['LOF_Score'] == -1).astype(int)

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')  
oc_svm.fit(X_df)

labels_ocsvm = oc_svm.predict(X_df)  
rfm_df['Churn_Label_OCSVM'] = (labels_ocsvm == -1).astype(int)

# Treatment feature
rfm_df['Treatment'] = (rfm_df['USED_EXTRA_DISCOUNT_RATIO'] > 0.05).astype(bool)

# ========================
# 3. UPLIFT MODELLING
# ========================

models = {
    "TwoModels_XGB_vs_RF": TwoModels(
        estimator_trmnt=XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, eval_metric='logloss', random_state=42),
        estimator_ctrl=RandomForestClassifier(n_estimators=1000, random_state=42),
        method='vanilla'
    ),
      'tm_logreg_svm': TwoModels(
        estimator_trmnt=LogisticRegression(max_iter=1000),
        estimator_ctrl=SVC(probability=True),
        method='vanilla'
    ),
      'ct_xgb': ClassTransformation(
        estimator=XGBClassifier( eval_metric='logloss')
    ),
    'ct_rf': ClassTransformation(
        estimator=RandomForestClassifier(n_estimators=1000, random_state=42),
    ),
}

# ========================
# 4. EVALUATING MODELS
# ========================

# Isolation Forest Label + Uplift Model
yLabel_IsoForest = rfm_df.loc[X_df.index, 'Churn_Label_IsoForest']
treatment = rfm_df.loc[X_df.index, 'Treatment'].astype(int)

X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X_df, yLabel_IsoForest, treatment, test_size=0.3, stratify=treatment, random_state=42)

# Initialize results dictionary
results = {}
best_score  = -1
best_model = None

# Experiment with different models
print("\n Isolation Forest Label + Uplift Model...")

for name, model in models.items():
    print(f"\n {name}...")

    model.fit(X_train, y_train, treat_train)
    preds = model.predict(X_test)
    
    qini_score = qini_auc_score(y_test, preds, treat_test)
    auuc_score = uplift_auc_score(y_test, preds, treat_test)
    results[name] = {
        "uplift": preds,
        "qini_auc": qini_score,
         "auuc": auuc_score
    }
    print(f"\n Qini AUC: {qini_score:.4f} \n AUUC: {auuc_score:.4f} ")
    if qini_score > best_score:
        best_score = qini_score
        best_model = model

# Local Outlier Factor Label + Uplift Model
yLabel_LOF = rfm_df.loc[X_df.index, 'Churn_Label_LOF']
treatment = rfm_df.loc[X_df.index, 'Treatment'].astype(int)

X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X_df, yLabel_LOF, treatment, test_size=0.3, stratify=treatment, random_state=42
)

# Experiment with different models
print("\n Local Outlier Factor Label + Uplift Model...")

for name, model in models.items():
    print(f"\n {name}...")

    model.fit(X_train, y_train, treat_train)
    preds = model.predict(X_test)
    
    qini_score = qini_auc_score(y_test, preds, treat_test)
    auuc_score = uplift_auc_score(y_test, preds, treat_test)
    results[name] = {
        "uplift": preds,
        "qini_auc": qini_score,
         "auuc": auuc_score,
    }
    print(f"\n Qini AUC: {qini_score:.4f} \n AUUC: {auuc_score:.4f} ")
    if qini_score > best_score:
        best_score = qini_score
        best_model = model

# One-Class SVM Label + Uplift Model
yLabel_OCSVM = rfm_df.loc[X_df.index, 'Churn_Label_OCSVM']
treatment = rfm_df.loc[X_df.index, 'Treatment'].astype(int)

X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X_df, yLabel_OCSVM, treatment, test_size=0.3, stratify=treatment, random_state=42
)

# Experiment with different models
print("\n One-Class SVM Label + Uplift Model...")

for name, model in models.items():
    print(f"\n {name}...")

    model.fit(X_train, y_train, treat_train)
    preds = model.predict(X_test)
    
    qini_score = qini_auc_score(y_test, preds, treat_test)
    auuc_score = uplift_auc_score(y_test, preds, treat_test)
    upliftk=uplift_at_k(y_test, preds, treat_test,'by_group',k=0.05)
    results[name] = {
        "uplift": preds,
        "qini_auc": qini_score,
         "auuc": auuc_score
    }
    print(f"\n Qini AUC: {qini_score:.4f} \n AUUC: {auuc_score:.4f} ")
    if qini_score > best_score:
        best_score = qini_score
        best_model = model

# ========================
# 5. PLOTTING RESULTS
# ========================

# Qini Curve
yLabel_OCSVM = rfm_df.loc[X_df.index, 'Churn_Label_OCSVM']
treatment = rfm_df.loc[X_df.index, 'Treatment'].astype(int)

X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X_df, yLabel_OCSVM, treatment, test_size=0.3, stratify=treatment, random_state=42
)
m=models['TwoModels_XGB_vs_RF'].fit(X_train, y_train, treat_train)
preds = m.predict(X_test)

x_qini, y_qini = qini_curve(y_true=y_test, uplift=preds, treatment=treat_test)

print ("Quini Curve Data:")

plt.figure(figsize=(8, 6))
plt.plot(x_qini, y_qini, label='Qini Curve')
plt.xlabel('Number of Targeted Individuals')
plt.ylabel('Incremental Responders')
plt.title('Qini Curve')
plt.legend()
plt.grid(True)
plt.show()

# Uplift Curve
x_uplift, y_uplift = uplift_curve(y_true=y_test, uplift=preds, treatment=treat_test)
print ("Uplift Curve Data:")

plt.figure(figsize=(8, 6))
plt.plot(x_uplift, y_uplift, label=f'Uplift Curve', linewidth=2)
plt.xlabel('Fraction of Targeted Individuals')
plt.ylabel('Cumulative Incremental Responders')
plt.title(f'Uplift Curve for Best Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Plot of Results
print("Bar Comparison of Qini AUC and AUUC across Models and Churn Labels")

data = {
    "Model": [
        "TwoModels_XGB_vs_RF", "TwoModels_LOGREG_vs_SVM ", "CT_XGB ", "CT_RF ",
        "TwoModels_XGB_vs_RF", "TwoModels_LOGREG_vs_SVM ", "CT_XGB ", "CT_RF ",
        "TwoModels_XGB_vs_RF", "TwoModels_LOGREG_vs_SVM ", "CT_XGB ", "CT_RF "
    ],
    "Churn_Label": [
        "OCSVM", "OCSVM", "OCSVM", "OCSVM",
        "Isolation Forest", "Isolation Forest", "Isolation Forest", "Isolation Forest",
        "LOF", "LOF", "LOF", "LOF"
    ],
    "Qini_AUC": [
        0.6018, 0.1768, 0.4571, 0.3520,
        0.5479, 0.4575, 0.3598, 0.3461,
        0.4405, -0.2836, 0.2001, 0.2410
    ],
    "AUUC": [
        0.4273, 0.0568, 0.5309, 0.5330,
        0.3153, 0.1481, 0.5000, 0.4011,
        0.2299, -0.0461, 0.3465, 0.5332
    ]
}

df = pd.DataFrame(data)

fig, axes = plt.subplots(2, 1, figsize=(14, 18))
metrics = ['Qini_AUC', 'AUUC']
benchmark_lines = [0.4,0.4]  

for i, metric in enumerate(metrics):
    sns.barplot(data=df, x='Model', y=metric, hue='Churn_Label', ax=axes[i], palette='Set2')
    axes[i].set_title(f'Figure {i+1}: {metric} across Models and Churn Labels', fontsize=15)
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].axhline(benchmark_lines[i], color='red', linestyle='--', label='Benchmark Threshold')
    axes[i].legend()

plt.tight_layout()
plt.show()

# ========================
# 6. TARGETS DATASET
# ========================

rfm_df.loc[X_test.index, 'Uplift_Score'] = preds

churned_customers = rfm_df[rfm_df['Churn_Label_OCSVM'] == 1]

top_churned_customers = churned_customers.sort_values('Uplift_Score', ascending=False).head(10000).copy()

def assign_discount(uplift):
    if uplift > 0.5:        
        return 0.25
    elif uplift > 0.3:      
        return 0.15
    elif uplift > 0.15:     
        return 0.05
    else:
        return 0.00          

top_churned_customers['Assigned_Discount'] = top_churned_customers['Uplift_Score'].apply(assign_discount)

top_churned_customers['Assigned_Discount'] = top_churned_customers['Uplift_Score'].apply(assign_discount)
top_churned_customers['Expected_Revenue_Saved'] = top_churned_customers['Monetary'] * top_churned_customers['Uplift_Score']
top_churned_customers['Discount_Cost'] = top_churned_customers['Monetary'] * top_churned_customers['Assigned_Discount']
top_churned_customers['ROI'] = top_churned_customers['Expected_Revenue_Saved'] / top_churned_customers['Discount_Cost'].replace(0, 1)

top_churned_customers['Uplift_Decile'] = pd.qcut(top_churned_customers['Uplift_Score'], 10, labels=False)

print("\nTop 10 Churned Customers with Assigned Discounts and ROI:")
print(top_churned_customers.head(10))

# ========================
# 8. SUMMARY OF TARGETED GROUP
# ========================
print("\nSummary of Targeted Group:")
print("Total Customers Targeted:", len(top_churned_customers))
print("Total Expected Revenue Saved:", round(top_churned_customers['Expected_Revenue_Saved'].sum(), 2))
print("Total Discount Cost:", round(top_churned_customers['Discount_Cost'].sum(), 2))
print("Mean ROI:", round(top_churned_customers['ROI'].mean(), 2))
print("Percentage of Customers with ROI > 1:", round((top_churned_customers['ROI'] > 1).mean() * 100, 2), "%")
print("Number of Customers with ROI > 1:", (top_churned_customers['ROI'] > 1).sum())
print("Targeted Group Summary:")
print("Mean ROI:", round(top_churned_customers['ROI'].mean(), 2))
print("ROI > 1 Count:", (top_churned_customers['ROI'] > 1).mean()*100)
print("Total Expected Revenue Saved:", round(top_churned_customers['Expected_Revenue_Saved'].sum(), 2))
print("Total Discount Cost:", round(top_churned_customers['Discount_Cost'].sum(), 2))