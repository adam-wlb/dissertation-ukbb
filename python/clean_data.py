"""clean_data.py - Cleans the biobank dataset ready for fitting classifiers"""
__author__ = "Adam Barron - 160212899"

import pandas as pd
from pandas_profiling import ProfileReport

# Load features extracted by GGIR
features = pd.read_csv('resources/part5_personsummary_WW_L30M100V400_T5A5.csv')
# Load class labels
class_labels = pd.read_csv('resources/simplified-subset.csv')
# Merge to add class labels
biobank = features.merge(class_labels, on='id_pla')
# Print data shape
print(biobank.shape)

# Produce data report with Pandas Reporting
report = ProfileReport(biobank, title='Biobank', html={'style':{'full_width':True}},minimal=True)
report.to_file(output_file='resources/biobank-pandas-profiling-report.html')

# Drop unneeded metadata columns
biobank.drop(['filename', 'id_pla', 'startday', 'calendardate'], axis=1, inplace=True)
# Drop constant columns
biobank.drop(biobank.loc[:, biobank.nunique() == 1], axis=1, inplace=True)
# Drop those columns with more than 10% NaN values
biobank.drop(biobank.loc[:, (biobank.isna().sum()) > (biobank.count() * 0.1)], axis=1, inplace=True)

# Sort features by skewness
features = []
skew = []
for col, vals in biobank.iteritems():
    features.append(col)
    skew.append(abs(vals.skew()))
skews_df = pd.DataFrame(list(zip(features, skew)), columns=['Feature', 'Skew']).sort_values(by='Skew', ascending=False)
print(skews_df)
skews_df.to_csv('resources/features_skew.csv', index=False)

# Drop highly skewed columns
biobank.drop(biobank.loc[:, (abs(biobank.skew()) > 30)], axis=1, inplace=True)

# Produce new data report on cleaned data
report = ProfileReport(biobank, title='Biobank-Cleaned', html={'style':{'full_width':True}}, minimal=True)
report.to_file(output_file='resources/biobank-pandas-profiling-report-cleaned.html')

# Export cleaned data as new csv
biobank.to_csv('resources/biobank.csv')
