"""
Baixa sessoes de F1 para o cache local (f1_cache/).

IMPORTANTE: rode este script na SUA MAQUINA LOCAL, nao no Codespace.
O servidor da F1 (CloudFront) bloqueia IPs de datacenter/nuvem com erro 403,
entao o download so funciona a partir de uma conexao residencial.

Uso:
    pip install fastf1
    python baixar_cache_f1.py

Depois de rodar, faca commit da pasta f1_cache/ (ou copie ela) para conseguir
trabalhar com os dados dentro do Codespace, sem precisar de internet.
"""

import os
import fastf1

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Ajuste aqui quais temporadas e corridas voce quer baixar.
ANOS = [2024]
TIPO_SESSAO = 'R'  # 'R' = corrida, 'Q' = classificacao, 'FP1'/'FP2'/'FP3' = treinos


def baixar_temporada(ano, tipo=TIPO_SESSAO):
    calendario = fastf1.get_event_schedule(ano, include_testing=False)

    for _, evento in calendario.iterrows():
        nome = evento['EventName']
        rodada = evento['RoundNumber']

        try:
            sessao = fastf1.get_session(ano, rodada, tipo)
            sessao.load()  # baixa e grava no cache
            print(f'[ok] {ano} R{rodada:02d} {nome} -> {len(sessao.laps)} voltas')
        except Exception as e:
            print(f'[falhou] {ano} R{rodada:02d} {nome}: {type(e).__name__}: {e}')


if __name__ == '__main__':
    for ano in ANOS:
        print(f'\n=== Baixando temporada {ano} ===')
        baixar_temporada(ano)

    print(f'\nCache salvo em: {CACHE_DIR}')
