"""
try to use sdv pacage to creat GAN
"""

import pandas as pd
from sdv.datasets.demo import download_demo
from sdv.lite import SingleTablePreset
#from sdv.tabular import CopulaGAN

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests')

synthesizer = SingleTablePreset(metadata, name='FAST_ML')
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=500)
print()
from sdv.evaluation.single_table import run_diagnostic

diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)