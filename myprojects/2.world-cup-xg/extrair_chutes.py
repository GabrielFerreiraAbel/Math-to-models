"""
Le os eventos brutos do StatsBomb no cache e monta UMA tabela de chutes,
salva em chutes_wc22.parquet.

O script so desaninha o JSON. Ele nao calcula distancia, angulo, nem decide
nada sobre penaltis -- isso e modelagem e fica no notebook.

Uso:
    python extrair_chutes.py
"""

import glob
import json
import os

import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(PASTA, 'statsbomb_cache')
SAIDA = os.path.join(PASTA, 'chutes_wc22.parquet')

# campos dentro de [shot] que sao dicionarios {'id':..., 'name':...}
SHOT_NOMEADOS = ['type', 'outcome', 'technique', 'body_part']

# flags booleanas: quando ausentes significam False, nao "faltando"
SHOT_FLAGS = ['first_time', 'one_on_one', 'aerial_won', 'open_goal',
              'deflected', 'follows_dribble', 'saved_to_post']

# campos da raiz do evento que sao dicionarios
RAIZ_NOMEADOS = ['player', 'team', 'possession_team', 'play_pattern', 'position']


def extrair_chute(e, match_id):
    s = e['shot']
    linha = {
        'match_id': match_id,
        'shot_id': e['id'],
        'period': e['period'],
        'minute': e['minute'],
        'second': e['second'],
        # location vem como [x, y] no campo de 120 x 80
        'x': e['location'][0],
        'y': e['location'][1],
        'duration': e.get('duration'),
        # under_pressure so aparece quando True
        'under_pressure': bool(e.get('under_pressure', False)),
        # benchmark do StatsBomb: NAO usar como feature, so para comparar no fim
        'statsbomb_xg': s['statsbomb_xg'],
    }

    for c in RAIZ_NOMEADOS:
        linha[c] = e[c]['name'] if c in e else None

    for c in SHOT_NOMEADOS:
        linha['shot_' + c] = s[c]['name'] if c in s else None

    for c in SHOT_FLAGS:
        linha[c] = bool(s.get(c, False))

    # alvo
    linha['gol'] = int(s['outcome']['name'] == 'Goal')

    return linha


def main():
    arquivos = sorted(glob.glob(os.path.join(CACHE, 'events_*.json')))
    if not arquivos:
        print(f'nenhum events_*.json em {CACHE}')
        return

    linhas = []
    for caminho in arquivos:
        match_id = int(os.path.basename(caminho).replace('events_', '').replace('.json', ''))
        with open(caminho, encoding='utf-8') as f:
            eventos = json.load(f)

        chutes = [e for e in eventos if e['type']['name'] == 'Shot']
        linhas.extend(extrair_chute(e, match_id) for e in chutes)
        print(f'[ok] {match_id} -> {len(chutes)} chutes')

    df = pd.DataFrame(linhas)
    df.to_parquet(SAIDA, index=False)

    mb = os.path.getsize(SAIDA) / 1024 / 1024
    print(f'\n{len(df)} chutes, {len(df.columns)} colunas')
    print(f'gols: {df.gol.sum()} ({df.gol.mean():.1%})')
    print(f'salvo em {SAIDA} ({mb:.2f} MB)')


if __name__ == '__main__':
    main()
