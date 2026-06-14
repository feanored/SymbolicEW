import numpy as np
import pandas as pd

_PATH_ = "C:/Users/eduar/Downloads/"

def main():
    dados1 = pd.read_csv(f"{_PATH_}ariel_completo_v1.csv.gz", compression="gzip")
    dados2 = pd.read_csv(f"{_PATH_}ariel_completo_v2.csv.gz", compression="gzip")
    print(dados1.columns, "\n", dados1.shape)
    print(dados2.columns, "\n", dados2.shape)

    # Gerando chave primária numérica
    dados1['fileid'] = dados1['file'].str.replace('-', '').astype(int)
    dados2['fileid'] = dados2['file'].str.replace('-', '').astype(int)

    # Conferindo se é chave primária -> OK
    print(len(dados1['fileid']), len(dados1['fileid'].unique()))
    print(len(dados2['fileid']), len(dados2['fileid'].unique()))

    dados1["mass_log10"] = np.log10(dados1["mass"])

    dados_raw = pd.merge(
        dados1[['fileid', 'file', 'RA', 'Dec', 'z', 'atflux', 'atmass', 'aZflux', 'aZmass', 'mass_log10', 'Av']],
        dados2[['fileid', 'oiii_5007_flux', 'oiii_5007_flux_err',
                'oiii_5007_ew', 'oiii_5007_ew_err', 'nii_6584_flux',
                'nii_6584_flux_err', 'nii_6584_ew', 'nii_6584_ew_err', 'halpha_flux',
                'halpha_flux_err', 'halpha_ew', 'halpha_ew_err', 'hbeta_flux',
                'hbeta_flux_err', 'hbeta_ew', 'hbeta_ew_err', 'oii_3727_flux',
                'oii_3727_flux_err', 'oii_3727_ew', 'oii_3727_ew_err']],
        on='fileid')
    print(dados_raw.columns, "\n", dados_raw.shape)

    dados_raw.to_csv(f"{_PATH_}ariel_completo.csv.gz", compression="gzip", index=False)

main()
