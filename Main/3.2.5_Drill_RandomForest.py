#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', 999)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
# Replace the path with the correct path for your data.
y2015 = pd.read_csv(
    "../Datasets/LoanStats3d.csv",
    skipinitialspace=True,
    header=1
)

# Note the warning about dtypes.


#%%
y2015.head()


#%%
# Remove two summary rows at the end that don't actually contain data.
y2015 = y2015[:-2]


#%%
# Convert ID and Interest Rate to numeric.
y2015['id'] = pd.to_numeric(y2015['id'], errors='coerce')
y2015['int_rate'] = pd.to_numeric(y2015['int_rate'].str.strip('%'), errors='coerce')

# Drop other columns with many unique variables
y2015.drop(['url', 'emp_title', 'zip_code', 'earliest_cr_line', 'revol_util',
            'sub_grade', 'addr_state', 'desc'], 1, inplace=True)


#%%
y2015.describe()

#%% [markdown]
# ## Reference model

#%%
from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

score = cross_val_score(rfc, X, Y, cv=10)
print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

#%% [markdown]
# ## DRILL: Third Attempt
# 
# So here's your task. Get rid of as much data as possible without dropping below an average of 90% accuracy in a 10-fold cross validation.
# 
# You'll want to do a few things in this process. First, dive into the data that we have and see which features are most important. This can be the raw features or the generated dummies. You may want to use PCA or correlation matrices.
# 
# Can you do it without using anything related to payment amount or outstanding principal? How do you know?

#%%
col_y = y2015['loan_status']
y2015.drop(labels=['loan_status'], axis=1,inplace = True)
y2015.insert(0, 'loan_status', col_y)
df_dumm = pd.get_dummies(y2015)
df_dumm = df_dumm.dropna(axis=1)


#%%
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df_dumm, 100).to_string())


#%%
# Your code here.

rfc = ensemble.RandomForestClassifier()
X = y2015.drop([ 
               "loan_status", # is Y 
               "id",
               "member_id",
               "loan_amnt",
               "funded_amnt",
               "funded_amnt_inv",
               "term",
               "int_rate",
               "installment",
               "grade",
               "emp_length",
               "home_ownership",
               "annual_inc",
               "verification_status",
               "issue_d",
               "pymnt_plan",
               "purpose",
               "title",
               "dti",
               "delinq_2yrs",
               "inq_last_6mths",
               "mths_since_last_delinq",
               "mths_since_last_record",
               "open_acc",
               "pub_rec",
               "revol_bal",
               "total_acc",
               "initial_list_status",
               #"out_prncp",
               #"out_prncp_inv",
               "total_pymnt",
               "total_pymnt_inv",
               "total_rec_prncp",
               "total_rec_int",
               "total_rec_late_fee",
               #"recoveries",
               "collection_recovery_fee",
               "last_pymnt_d",
               "last_pymnt_amnt",
               "next_pymnt_d",
               "last_credit_pull_d",
               "collections_12_mths_ex_med",
               "mths_since_last_major_derog",
               "policy_code",
               "application_type",
               "annual_inc_joint",
               "dti_joint",
               "verification_status_joint",
               "acc_now_delinq",
               "tot_coll_amt",
               "tot_cur_bal",
               "open_acc_6m",
               "open_act_il",
               "open_il_12m",
               "open_il_24m",
               "mths_since_rcnt_il",
               "total_bal_il",
               "il_util",
               "open_rv_12m",
               "open_rv_24m",
               "max_bal_bc",
               "all_util",
               "total_rev_hi_lim",
               "inq_fi",
               "total_cu_tl",
               "inq_last_12m",
               "acc_open_past_24mths",
               "avg_cur_bal","bc_open_to_buy",
               "bc_util",
               "chargeoff_within_12_mths",
               "delinq_amnt",
               "mo_sin_old_il_acct",
               "mo_sin_old_rev_tl_op",
               "mo_sin_rcnt_rev_tl_op",
               "mo_sin_rcnt_tl",
               "mort_acc",
               "mths_since_recent_bc",
               "mths_since_recent_bc_dlq",
               "mths_since_recent_inq",
               "mths_since_recent_revol_delinq",
               "num_accts_ever_120_pd",
               "num_actv_bc_tl","num_actv_rev_tl",
               "num_bc_sats",
               "num_bc_tl","num_il_tl",
               "num_op_rev_tl",
               "num_rev_accts",
               "num_rev_tl_bal_gt_0",
               "num_sats",
               "num_tl_120dpd_2m",
               "num_tl_30dpd",
               "num_tl_90g_dpd_24m",
               "num_tl_op_past_12m",
               "pct_tl_nvr_dlq",
               "percent_bc_gt_75","pub_rec_bankruptcies",
               "tax_liens","tot_hi_cred_lim",
               "total_bal_ex_mort",
               "total_bc_limit",
               "total_il_high_credit_limit",
               "revol_bal_joint",
               "sec_app_earliest_cr_line",
               "sec_app_inq_last_6mths",
               "sec_app_mort_acc","sec_app_open_acc",
               "sec_app_revol_util","sec_app_open_act_il",
               "sec_app_num_rev_accts",
               "sec_app_chargeoff_within_12_mths",
               "sec_app_collections_12_mths_ex_med",
               "sec_app_mths_since_last_major_derog",
               "hardship_flag",
               "hardship_type",
               "hardship_reason",
               "hardship_status",
               "deferral_term",
               "hardship_amount",
               "hardship_start_date",
               "hardship_end_date",
               "payment_plan_start_date",
               "hardship_length",
               "hardship_dpd",
               "hardship_loan_status",
               "orig_projected_additional_accrued_interest",
               "hardship_payoff_balance_amount",
               "hardship_last_payment_amount",
               "debt_settlement_flag",
               "debt_settlement_flag_date",
               "settlement_status",
               "settlement_date",
               "settlement_amount",
               "settlement_percentage",
               "settlement_term" ], 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

score = cross_val_score(rfc, X, Y, cv=10)
print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

#%% [markdown]
# ### Drill achieved : Accuracy: 0.940 (+/- 0.007)

