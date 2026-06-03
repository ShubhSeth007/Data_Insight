import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- 1. GLOBAL STYLE SETTING ---
# Giving the charts a clean, modern, high-contrast aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})
CUSTOM_PALETTE = ["#1A5276", "#E67E22", "#2ECC71", "#9B59B6", "#34495E"]
sns.set_palette(CUSTOM_PALETTE)

# --- 2. HELPER FUNCTIONS FOR ANALYSIS ---

def get_column_types(df):
    """Categorizes columns into numerical and categorical types."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return num_cols, cat_cols

def generate_missing_data_insights(df):
    """Generates text alerts for missing data."""
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    insights = []
    if high_missing.empty:
        insights.append("🎉 **Perfect Health:** Dataset has zero missing values!")
    else:
        for col, pct in high_missing.items():
            if pct > 10:
                insights.append(f"⚠️ **Critical Alert:** `{col}` is missing **{pct:.1f}%** of its data. Consider imputation or removal.")
            else:
                insights.append(f"💡 **Note:** `{col}` has minor missing data (**{pct:.1f}%**).")
    return insights

# --- 3. STREAMLIT UI SETUP ---
st.set_page_config(page_title="DataInsight EDA Tool", layout="wide", page_icon="📊")

st.title("📊 DataInsight: Automated EDA Dashboard")
st.markdown("Upload any CSV file to generate production-grade visualizations and statistical profiles instantly.")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    num_cols, cat_cols = get_column_types(df)
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🎯 Dashboard Settings")
    max_categories = st.sidebar.slider("Max categories to display in plots", 5, 20, 10)
    
    # --- TABS FOR ORGANIZED VIEW ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Numerical Features", "📊 Categorical Features", "🔗 Relationships & Correlations"])
    
    # ==========================================
    # TAB 1: DATA OVERVIEW
    # ==========================================
    with tab1:
        st.header("Dataset Overview")
        
        # High-level Metrics Card
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Numerical Features", len(num_cols))
        col4.metric("Categorical Features", len(cat_cols))
        
        st.markdown("### Preview Data (First 5 Rows)")
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("### Missing Data Profiling")
        m_col1, m_col2 = st.columns([1, 1])
        
        with m_col1:
            missing_insights = generate_missing_data_insights(df)
            for insight in missing_insights:
                st.markdown(insight)
                
        with m_col2:
            if df.isnull().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
                ax.set_title("Missing Data Matrix (Yellow = Missing)")
                st.pyplot(fig)
            else:
                st.success("No missing data visualization needed!")

    # ==========================================
    # TAB 2: NUMERICAL FEATURES
    # ==========================================
    with tab2:
        st.header("Numerical Distributions & Outliers")
        if not num_cols:
            st.info("No numerical columns found in the dataset.")
        else:
            selected_num = st.selectbox("Select a numerical column to analyze:", num_cols)
            
            # Statistical calculations
            col_data = df[selected_num].dropna()
            skewness = col_data.skew()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution Plot
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(col_data, kde=True, color=CUSTOM_PALETTE[0], ax=ax)
                sns.despine()
                ax.set_title(f"Distribution of {selected_num}")
                st.pyplot(fig)
                
                # Distribution Insight
                if abs(skewness) > 1:
                    st.warning(f"💡 **Insight:** `{selected_num}` is highly skewed (Skewness: **{skewness:.2f}**). Consider a log transformation if using linear algorithms.")
                else:
                    st.success(f"💡 **Insight:** `{selected_num}` is relatively symmetric (Skewness: **{skewness:.2f}**).")
                    
            with col2:
                # Boxplot for Outliers
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=col_data, color=CUSTOM_PALETTE[1], ax=ax)
                sns.despine()
                ax.set_title(f"Box Plot / Outlier Check for {selected_num}")
                st.pyplot(fig)
                
                # IQR Outlier calculation
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                outliers = col_data[(col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))]
                
                if not outliers.empty:
                    st.error(f"⚠️ **Outlier Alert:** Found **{len(outliers)}** data points acting as statistical outliers out of {len(col_data)} entries.")
                else:
                    st.success("🎉 **Clean Feature:** No statistical outliers detected using standard IQR boundary limits.")

    # ==========================================
    # TAB 3: CATEGORICAL FEATURES
    # ==========================================
    with tab3:
        st.header("Categorical Value Frequencies")
        if not cat_cols:
            st.info("No categorical columns found in the dataset.")
        else:
            selected_cat = st.selectbox("Select a categorical column to analyze:", cat_cols)
            
            cat_counts = df[selected_cat].value_counts()
            
            # Handle high cardinality gracefully
            if len(cat_counts) > max_categories:
                st.info(f"Showing top {max_categories} out of {len(cat_counts)} unique categories.")
                cat_counts = cat_counts.head(max_categories)
                
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis", ax=ax, orient='h')
            sns.despine()
            ax.set_title(f"Top {len(cat_counts)} Frequency Distribution of {selected_cat}")
            ax.set_xlabel("Count")
            st.pyplot(fig)
            
            # Insights
            top_cat = cat_counts.index[0]
            top_pct = (cat_counts.values[0] / len(df)) * 100
            st.markdown(f"💡 **Insight:** The dominant class is **'{top_cat}'**, capturing **{top_pct:.1f}%** of all observed items in this attribute.")

    # ==========================================
    # TAB 4: RELATIONSHIPS & CORRELATIONS
    # ==========================================
    with tab4:
        st.header("Correlation Matrices & Feature Interdependence")
        
        # 1. Correlation Matrix Heatmap
        if len(num_cols) >= 2:
            st.subheader("Numerical Correlation Heatmap")
            corr = df[num_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            mask = np.triu(np.ones_like(corr, dtype=bool)) # clean upper triangle look
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=.5, ax=ax)
            st.pyplot(fig)
            
            # Extract highly correlated features automatically
            high_corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                        
            if high_corr_pairs:
                st.warning("⚠️ **Multicollinearity Flag:** The following feature pairings show strong correlation ($|r| > 0.7$):")
                for col1, col2, val in high_corr_pairs:
                    st.markdown(f"* `{col1}` & `{col2}` $\\rightarrow$ **r = {val:.2f}**")
            else:
                st.success("🎉 **A-OK Feature Independence:** No severe overlapping correlation threats discovered across features.")
        else:
            st.info("Need at least two numerical columns to evaluate correlation structures.")
            
        st.write("---")
        
        # 2. Bivariate Scatter Analysis
        st.subheader("Bivariate Value Mapping (Scatter Visualizer)")
        if len(num_cols) >= 2:
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                x_axis = st.selectbox("Select Independent Variable (X):", num_cols, index=0)
            with sc_col2:
                # Safeguard index out of bounds if there's only 2 items
                y_idx = 1 if len(num_cols) > 1 else 0
                y_axis = st.selectbox("Select Dependent Variable (Y):", num_cols, index=y_idx)
                
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(data=df, x=x_axis, y=y_axis, color=CUSTOM_PALETTE[2], scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
            sns.despine()
            ax.set_title(f"Trend Intercept Map: {x_axis} vs {y_axis}")
            st.pyplot(fig)
        else:
            st.info("Insufficient continuous properties to evaluate bivariate relationships.")

else:
    # Landing page display if file not loaded yet
    st.info("👈 Please upload a target CSV data file in the sidebar to kickstart the auto-analysis workflow.")
    
    # Showcase mockup visual
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=800&q=80", caption="Ready to process data architectures", width=600)
