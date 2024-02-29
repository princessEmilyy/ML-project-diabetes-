"""
This is a demo for GAN model - Synthesize a Table (CTGAN)
"""

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality


# uploadig the data 

real_data = pd.read_csv("train_df_after_rows_filtration.csv")
real_data = real_data[real_data['readmitted']=='<30']
metadata_df = pd.read_csv("metadata.csv")
#metadata_df.pop("age")
metadata_df = metadata_df
metadata_dict = metadata_df.to_dict('r')
#metadata_obj = SingleTableMetadata.load_from_dict(metadata_dict)
real_data = real_data[list(metadata_df.columns)]#.fillna("N")
# create a metadata file 
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
for var in metadata_df:
    if metadata_df[var].iloc[0]== 'numerical':
        metadata.update_column(
            column_name=var,
            sdtype='numerical',
            computer_representation='Float')
    elif metadata_df[var].iloc[0]== 'id':
        metadata.update_column(
            column_name=var,
            sdtype='id',
            regex_format='[0-9]{9}')
    else:
        metadata.update_column(column_name=var,sdtype=metadata_df[var].iloc[0])
metadata.remove_primary_key()
metadata.set_primary_key(column_name='encounter_id')
metadata.validate_data(data=real_data)
# creat the new data
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(real_data)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.head()

# evaluating
diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

quality_report.get_details('Column Shapes')