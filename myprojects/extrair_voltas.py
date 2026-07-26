"""
Le as sessoes ja baixadas no cache e monta UM dataframe unico de voltas,
salvo em voltas_f1.parquet.

Rode DEPOIS do baixar_cache_f1.py, na mesma maquina (local).
O parquet gerado tem poucos MB e e ele que deve ir para o Git --
o cache bruto (f1_cache/) fica de fora.

Uso:
    pip install fastf1 pyarrow
    python extrair_voltas.py
"""

import os
import fastf1
import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PASTA, 'f1_cache')
SAIDA = os.path.join(PASTA, 'voltas_f1.parquet')

fastf1.Cache.enable_cache(CACHE_DIR)

ANOS = [2024]
TIPO_SESSAO = 'R'

# Colunas de tempo que viram segundos (float), mais faceis de modelar
COLUNAS_TEMPO = [
    'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
    'PitInTime', 'PitOutTime',
]

COLUNAS_VOLTA = [
    'Driver', 'DriverNumber', 'Team', 'LapNumber', 'Stint',
    'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
    'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
    'Compound', 'TyreLife', 'FreshTyre',
    'TrackStatus', 'Position', 'IsAccurate', 'Deleted',
    'PitInTime', 'PitOutTime', 'LapStartDate',
]

COLUNAS_CLIMA = [
    'AirTemp', 'TrackTemp', 'Humidity', 'Pressure',
    'Rainfall', 'WindSpeed', 'WindDirection',
]


def extrair_sessao(ano, rodada, nome_evento, tipo=TIPO_SESSAO):
    sessao = fastf1.get_session(ano, rodada, tipo)
    sessao.load()

    voltas = sessao.laps
    if voltas is None or len(voltas) == 0:
        raise ValueError('sessao sem voltas')

    df = voltas.reset_index(drop=True)

    # clima alinhado volta a volta
    try:
        clima = voltas.get_weather_data().reset_index(drop=True)
        for col in COLUNAS_CLIMA:
            if col in clima.columns:
                df[col] = clima[col]
    except Exception as e:
        print(f'    (sem clima: {type(e).__name__})')

    # so as colunas que interessam e que existem de fato
    cols = [c for c in COLUNAS_VOLTA + COLUNAS_CLIMA if c in df.columns]
    df = df[cols].copy()

    # timedelta -> segundos
    for col in COLUNAS_TEMPO:
        if col in df.columns:
            df[col] = df[col].dt.total_seconds()

    # identificacao do evento
    df.insert(0, 'Ano', ano)
    df.insert(1, 'Rodada', rodada)
    df.insert(2, 'Evento', nome_evento)
    df.insert(3, 'Sessao', tipo)

    return df


def main():
    partes = []

    for ano in ANOS:
        calendario = fastf1.get_event_schedule(ano, include_testing=False)

        for _, evento in calendario.iterrows():
            rodada = evento['RoundNumber']
            nome = evento['EventName']

            try:
                df = extrair_sessao(ano, rodada, nome)
                partes.append(df)
                print(f'[ok] {ano} R{rodada:02d} {nome} -> {len(df)} voltas')
            except Exception as e:
                print(f'[pulou] {ano} R{rodada:02d} {nome}: {type(e).__name__}: {e}')

    if not partes:
        print('\nNenhuma sessao extraida. O cache esta populado?')
        return

    completo = pd.concat(partes, ignore_index=True)
    completo.to_parquet(SAIDA, index=False)

    tamanho_mb = os.path.getsize(SAIDA) / 1024 / 1024
    print(f'\n{len(completo)} voltas, {len(completo.columns)} colunas')
    print(f'salvo em {SAIDA} ({tamanho_mb:.1f} MB)')


if __name__ == '__main__':
    main()
