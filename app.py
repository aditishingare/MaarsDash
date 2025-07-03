
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Marketing Analytics Suite", layout="wide")

# ---------- Data Upload/Load ----------
st.sidebar.header("🔄 Data Source")
default_path = "raj excel .xlsx"
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    df = pd.read_excel(default_path)

st.sidebar.success("Data loaded successfully!")

# Helper lists
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# ---------- TABS ----------
tabs = st.tabs([
    "📊 Data Visualisation", 
    "🧮 Classification", 
    "🌀 Clustering", 
    "🔗 Association Rules", 
    "📈 Regression"
])

# ========= 1. DATA VISUALISATION =========
with tabs[0]:
    st.header("📊 Descriptive Insights")

    # --- Column mapping for key metrics ---
    st.subheader("Map Columns for Pre‑built Charts")
    col1, col2, col3 = st.columns(3)
    with col1:
        channel_col = st.selectbox("Channel Column", cat_cols)
        campaign_col = st.selectbox("Campaign Type Column", cat_cols)
        location_col = st.selectbox("Location Column", cat_cols)
    with col2:
        conv_col = st.selectbox("Conversion Rate Column (numeric)", num_cols)
        roi_col  = st.selectbox("ROI Column (numeric)", num_cols)
        ctr_col  = st.selectbox("CTR Column (numeric)", num_cols)
    with col3:
        cpc_col  = st.selectbox("Cost per Click Column (numeric)", num_cols)
        acq_cost_col = st.selectbox("Acquisition Cost Column (numeric)", num_cols)
        duration_col = st.selectbox("Campaign Duration Column (numeric)", num_cols)

    custseg_col = st.selectbox("Customer Segment Column", cat_cols)
    target_col  = st.selectbox("Target Audience Column", cat_cols)
    clicks_col  = st.selectbox("Clicks Column (numeric)", num_cols)
    engage_col  = st.selectbox("Engagement Score Column (numeric)", num_cols)

    def _agg(data, gcol, vcol):
        return data[[gcol, vcol]].dropna().groupby(gcol, as_index=False).mean()

    # Generate 10 charts
    charts = [
        ("Channels vs Avg Conversion Rate", _agg(df, channel_col, conv_col), channel_col, conv_col, "bar"),
        ("Boxplot: Channel vs Conversion Rate", df[[channel_col, conv_col]].dropna(), channel_col, conv_col, "box"),
        ("Campaign Type vs Avg ROI", _agg(df, campaign_col, roi_col), campaign_col, roi_col, "bar"),
        ("Avg CPC by Customer Segment", _agg(df, custseg_col, cpc_col), custseg_col, cpc_col, "bar"),
        ("Target Audience vs Avg Clicks", _agg(df, target_col, clicks_col), target_col, clicks_col, "bar"),
        ("Channel vs Avg CTR", _agg(df, channel_col, ctr_col), channel_col, ctr_col, "bar"),
        ("Location vs Avg Acquisition Cost", _agg(df, location_col, acq_cost_col), location_col, acq_cost_col, "bar"),
        ("Customer Segment vs Avg Engagement", _agg(df, custseg_col, engage_col), custseg_col, engage_col, "bar"),
        ("Campaign Type vs Avg Duration Days", _agg(df, campaign_col, duration_col), campaign_col, duration_col, "bar"),
        ("ROI vs Duration Scatter", df[[roi_col, duration_col]].dropna(), roi_col, duration_col, "scatter")
    ]

    for title, d, xcol, ycol, kind in charts:
        st.subheader(title)
        if kind == "bar":
            chart = alt.Chart(d).mark_bar().encode(
                x=alt.X(f"{xcol}:N", title=xcol),
                y=alt.Y(f"{ycol}:Q", title=f"Average {ycol}"),
                tooltip=[xcol, alt.Tooltip(ycol, format=".2f")]
            ).properties(height=300)
        elif kind == "box":
            chart = alt.Chart(d).mark_boxplot().encode(
                x=alt.X(f"{xcol}:N", title=xcol),
                y=alt.Y(f"{ycol}:Q", title=ycol),
                tooltip=[xcol, ycol]
            ).properties(height=300)
        else:  # scatter
            chart = alt.Chart(d).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X(f"{xcol}:Q"),
                y=alt.Y(f"{ycol}:Q"),
                tooltip=[xcol, ycol]
            ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"**Insight:** The chart depicts how **{ycol}** varies across **{xcol}**.")

# ========= 2. CLASSIFICATION =========
with tabs[1]:
    st.header("🧮 Supervised Classification")

    target_var = st.selectbox("Select Target (categorical)", cat_cols)
    feature_vars = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target_var])

    test_size = st.slider("Test Size (fraction)", 0.1, 0.4, 0.25, 0.05)

    run_cls = st.button("Run Classification")
    if run_cls and feature_vars:
        X = pd.get_dummies(df[feature_vars], drop_first=True)
        y = df[target_var]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results = []
        y_proba_dict = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            results.append([name, acc, prec, rec, f1])

            # Probabilities for ROC
            if len(y.unique()) == 2:
                y_proba_dict[name] = model.predict_proba(X_test)[:,1]

        res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
        st.subheader("Performance Metrics")
        st.dataframe(res_df.style.format({c:"{:.2%}" for c in res_df.columns[1:]}), height=200)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm_model = st.selectbox("Choose Model", list(models.keys()))
        cm = confusion_matrix(y_test, models[cm_model].predict(X_test))
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
        ax.set_title(f"Confusion Matrix – {cm_model}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        st.pyplot(fig)

        # ROC Curves
        if len(y.unique()) == 2:
            st.subheader("ROC Curves")
            fig2, ax2 = plt.subplots()
            for name, proba in y_proba_dict.items():
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = roc_auc_score(y_test, proba)
                ax2.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
            ax2.plot([0,1],[0,1],"--", label="Random")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve Comparison")
            ax2.legend()
            st.pyplot(fig2)

        # Upload new data for prediction
        st.subheader("Predict on New Data")
        new_file = st.file_uploader("Upload new feature data (CSV/XLSX)", key="pred_upload")
        if new_file:
            new_df = pd.read_csv(new_file) if new_file.name.endswith("csv") else pd.read_excel(new_file)
            new_X = pd.get_dummies(new_df[feature_vars], drop_first=True).reindex(columns=X.columns, fill_value=0)
            best_model_name = res_df.sort_values("F1", ascending=False).iloc[0]["Model"]
            preds = models[best_model_name].predict(new_X)
            new_df["Prediction"] = preds
            st.write(new_df.head())
            csv = new_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# ========= 3. CLUSTERING =========
with tabs[2]:
    st.header("🌀 Unsupervised Clustering (K‑Means)")

    cluster_features = st.multiselect("Select Numeric Features for Clustering", num_cols)
    if cluster_features:
        k_slider = st.slider("Number of Clusters (k)", 2, 10, 3, 1)
        if st.button("Run K‑Means"):
            X = df[cluster_features].dropna()
            distortions = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(X)
                distortions.append(km.inertia_)

            # Elbow chart
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(1,11), distortions, marker="o")
            ax_elbow.set_xlabel("k"); ax_elbow.set_ylabel("Inertia")
            ax_elbow.set_title("Elbow Chart for Optimal k")
            st.pyplot(fig_elbow)

            # Final model
            final_km = KMeans(n_clusters=k_slider, random_state=42)
            labels = final_km.fit_predict(X)
            df_clusters = X.copy()
            df_clusters["Cluster"] = labels
            st.subheader("Cluster Summary (mean values)")
            st.dataframe(df_clusters.groupby("Cluster").mean())

            # PCA for 2‑D visualization
            if len(cluster_features) >= 2:
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X)
                comp_df = pd.DataFrame(comps, columns=["PC1","PC2"])
                comp_df["Cluster"] = labels
                chart = alt.Chart(comp_df).mark_circle(size=60, opacity=0.6).encode(
                    x="PC1:Q", y="PC2:Q", color="Cluster:N", tooltip=["Cluster"]
                ).properties(title="PCA Projection of Clusters", height=400)
                st.altair_chart(chart, use_container_width=True)

            # Download labelled data
            out_df = df.copy()
            out_df["Cluster"] = np.nan
            out_df.loc[X.index, "Cluster"] = labels
            csv_clusters = out_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Cluster‑labelled Data",
                data=csv_clusters,
                file_name="clustered_data.csv",
                mime="text/csv"
            )

# ========= 4. ASSOCIATION RULES =========
with tabs[3]:
    st.header("🔗 Market Basket – Association Rules")

    assoc_cols = st.multiselect("Select Categorical Columns", cat_cols)
    if len(assoc_cols) >= 2:
        min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
        if st.button("Generate Rules"):
            basket = df[assoc_cols].astype(str)
            one_hot = pd.get_dummies(basket)
            frequent = apriori(one_hot, min_support=min_sup, use_colnames=True)
            rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
            top_rules = rules.sort_values("confidence", ascending=False).head(10)
            st.dataframe(top_rules[["antecedents","consequents","support","confidence","lift"]])

# ========= 5. REGRESSION =========
with tabs[4]:
    st.header("📈 Regression Insights")

    reg_target = st.selectbox("Select Target (numeric)", num_cols)
    reg_features = st.multiselect("Select Feature Columns", [c for c in num_cols if c != reg_target])

    if st.button("Run Regression") and reg_features:
        X = df[reg_features].fillna(df[reg_features].median())
        y = df[reg_target].fillna(df[reg_target].median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        regr_models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }

        reg_results = []
        preds_dict = {}
        for name, model in regr_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds_dict[name] = y_pred
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            reg_results.append([name, mae, mse, r2])

        reg_df = pd.DataFrame(reg_results, columns=["Model","MAE","MSE","R²"])
        st.dataframe(reg_df.style.format({"MAE":"{:.2f}", "MSE":"{:.2f}", "R²":"{:.2f}"}))

        # Scatter for best model (highest R²)
        best_model = reg_df.sort_values("R²", ascending=False).iloc[0]["Model"]
        st.subheader(f"Predicted vs Actual – {best_model}")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, preds_dict[best_model], alpha=0.6)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
        ax3.set_xlabel("Actual"); ax3.set_ylabel("Predicted")
        st.pyplot(fig3)
