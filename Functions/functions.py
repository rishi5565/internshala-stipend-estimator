import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def univarstats(df):
    output_df = pd.DataFrame(columns=["Dtype", "Numeric", "Count", "Missing", "Unique", "Mode", "Mean", "Min", "25%", "Median", "75%", "Max", "Std", "Skew", "Kurt"])
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            output_df.loc[col] = [df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].mode().values[0], df[col].mean(),
             df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75), df[col].max(), df[col].std(), df[col].skew(), df[col].kurt()]
        
        else:
            output_df.loc[col] = [df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].mode().values[0], "-", "-", "-", "-",
            "-", "-", "-", "-", "-"]

    return output_df.sort_values("Unique", ascending=True).sort_values(by=["Numeric", "Skew"], ascending=False)

def anova(df, feature, label):
    from scipy import stats

    groups = df[feature].unique()
    df_grouped = df.groupby(feature)
    group_labels = []
    for g in groups:
        g_list = df_grouped.get_group(g)
        group_labels.append(g_list[label])

    return stats.f_oneway(*group_labels)

def heteroscedasticity(df, feature, label):
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.formula.api import ols

    model = ols(formula=(label + "~" + feature), data=df).fit()

    output_df = pd.DataFrame(columns=["-------LM stat", "      LM p-value", "  F-stat", "  F p-value"])

    try:
        white_test = het_white(model.resid, model.model.exog)
        output_df.loc["White"] = white_test
    except:
        print("Unable to run White test of heteroscedasticity")

    bp_test = het_breuschpagan(model.resid, model.model.exog)
    output_df.loc["B-Pg"] = bp_test
    
    return output_df.round(3)

def scatter(feature, label):
    from scipy import stats

    m, b, r, p, err = stats.linregress(feature, label)
    
    textstr = "y = " + str(round(m, 2)) + "x + " + str(round(b, 2)) + "\n"
    textstr += "r2 = " + str(round(r**2, 2)) + "\n"
    textstr += "p = " + str(p) + "\n"
    textstr += str(feature.name) + " skew = " + str(round(feature.skew(), 2)) + "\n"
    textstr += str(label.name) + " skew = " + str(round(label.skew(), 2)) + "\n"
    textstr += str(heteroscedasticity(pd.DataFrame(label).join(pd.DataFrame(feature)), feature.name, label.name))

    sns.set(color_codes = True)
    ax = sns.jointplot(feature, label, kind="reg")
    ax.fig.text(1, 0.114, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()

def bar_chart(df, feature, label):
    oneway = anova(df, feature, label)

    textstr = "              ANOVA" + "\n"
    textstr += "F:          " + str(oneway[0].round(2)) + "\n"
    textstr += "p-value:          " + str(oneway[1]) + "\n\n"

    ax = sns.barplot(df[feature], df[label])
    ax.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()


def bivarstats(df, label):

    from scipy import stats

    output_df = pd.DataFrame(columns=["Stat", "+/-", "Effect Size", "p-value"])

    for col in df:
        if not col == label:
            if df[col].isnull().sum() == 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    r, p = stats.pearsonr(df[label], df[col])
                    if r > 0:
                        output_df.loc[col] = ['r', 'Positive(+)', abs(round(r, 3)), p]
                    else:
                        output_df.loc[col] = ['r', 'Negative(-)', abs(round(r, 3)), p]
                    scatter(df[col],  df[label])
                else:
                    F, p = anova(df[[col, label]], col, label)
                    output_df.loc[col] = ["F", "", round(F, 3), p]
                    bar_chart(df, col, label)
            else:
                output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]
    return output_df.sort_values(["Stat", "Effect Size"], ascending=False)

def chi_sq(df, feature, label):
    from scipy.stats import chi2_contingency
    contingency = pd.crosstab(df[label], df[feature])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return(contingency, chi2, p, dof, expected)

def chistats(df, label):

    output_df = pd.DataFrame(columns=["Stat", "+/-", "Effect Size", "p-value"])

    for col in df:
        if not col == label:
            if df[col].isnull().sum() == 0:
                if (pd.api.types.is_string_dtype(df[col])):
                    contingency, chi2, p, dof, expected = chi_sq(df, col, label)
                    output_df.loc[col] = ["Chi-sq", "", round(chi2, 3), p]
                    print("\n-----------------------------------\n", contingency)
                    print("\n\nChi-Sq:" ,chi2, "\np:", p,"\nDOF:", dof,"\nExpected:\n\n", expected)

            else:
                output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]
    return output_df.sort_values(["Stat", "Effect Size"], ascending=False)


def vif(df):
    from sklearn.linear_model import LinearRegression

    vif_dict, tolerance_dict = {} , {}

    for col in df:
        X= df.drop(columns=[col])
        y = df[col]

        r_squared = LinearRegression().fit(X, y).score(X, y)

        if r_squared < 1:
            vif = 1 / (1 - r_squared)
        else:
            vif = 100
        vif_dict[col] = vif

        tolerance = 1 - r_squared
        tolerance_dict[col] = tolerance

        vif_output = pd.DataFrame({"VIF": vif_dict, "Tolerance": tolerance_dict})

    return vif_output.sort_values(by=["VIF"], ascending=False)