import pandas as pd

_PATH_ = "./"
_PATH_STAR_ = "/work2/laerte/BACKUP/laerte/ciencia/linhas/ariel/Vitor_Data/stellar+nebular/"
_PATH_DOBBY_ = "/work2/laerte/BACKUP/laerte/ciencia/linhas/ariel/Vitor_Data/dobby_output/"

def read_dobby(path):
    """Lê arquivo .txt do dobby pulando a linha separadora de traços."""
    return pd.read_csv(path, sep=r'\s+', skiprows=[1], engine='python')

def read_mod(path):
    """Lê arquivo .mod do starlight (wl, stellar+nebular, stellar, nebular)."""
    return pd.read_csv(
        path, sep=r'\s+', comment='#', header=None,
        names=['wl', 'stellar_nebular', 'stellar', 'nebular'],
        engine='python'
    )

def build_row(gal, dobby, star):
    """Monta um dict com todos os dados de dobby e janelas de lambdas do Starlight."""
    row = {'file': gal}

    for _, r in dobby.iterrows():
        line = r['line']
        center = r['lambda']  # comprimento-alvo

        # ---------- colunas do dobby ----------
        for col in dobby.columns:
            if col not in ['line', 'El_flag', 'El_lcrms', 'El_vdins']:
                col_name = f"{line}_{col}".replace('El_', '').replace('[', '').replace(']', '_')
                row[col_name] = r[col]

        # ---------- janela ±5 Å do starlight ----------
        mask = (star['wl'] >= center - 10) & (star['wl'] <= center + 10)
        window = star[mask]
        for col in ('wl', 'stellar_nebular', 'stellar', 'nebular'):
            col_name = f"{line}_star_{col}".replace('[', '').replace(']', '_')
            row[col_name] = window[col].tolist()

    return row

def main():
    dados = pd.read_csv("./ariel_completo.csv.gz", compression="gzip")

    rows = []
    n = 0
    for gal in dados["file"]:
        try:
            star  = read_mod(f"{_PATH_STAR_}{gal}.mod")
            dobby = read_dobby(f"{_PATH_DOBBY_}{gal}.txt")
            rows.append(build_row(gal, dobby, star))
            print("Found: ", gal)
            n += 1
        except FileNotFoundError:
            rows.append({'file': gal})

    extra = pd.DataFrame(rows)
    dados = dados.merge(extra, on='file', how='left')

    dados.to_parquet(f"{_PATH_}ariel_stardobby.parquet", index=False)
    print(f"A total of {n} galaxies were found!")

main()
