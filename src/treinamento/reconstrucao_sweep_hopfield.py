"""Módulo para reconstrução contínua de vetores SWeeP 600D com memória Hopfield completa do conjunto Pan."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.io as sio
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .hopfield import ModernHopfieldNetwork
from .projetor_sweep import ProjetorSWeePR

logger = logging.getLogger(__name__)

PathType = str | os.PathLike[str]

NOMES_CLASSES_PADRAO: dict[int, str] = {
    1: "Classe 1 (Excitatório)",
    2: "Classe 2 (Inibitório)",
    3: "Classe 3 (Astrócito)",
    4: "Classe 4 (Oligodendrócito)",
    5: "Classe 5 (OPC)",
    6: "Classe 6 (Micróglia)",
    7: "Classe 7 (Endotelial/Vascular)",
}


def substituir_colunas_faltantes_esparso(
    matriz: sp.spmatrix | NDArray[Any],
    indices_colunas: Sequence[int] | NDArray[Any],
    fill_value: float = 0.5,
) -> sp.csr_matrix:
    """Substitui colunas gênicas faltantes pelo valor sentinela preservando formato esparso CSR.

    Parameters
    ----------
    matriz : sp.spmatrix | NDArray
        Matriz de expressão gênica original (ex.: dataset Pan completo 58.326 × 61.541).
    indices_colunas : Sequence[int] | NDArray
        Índices das colunas a serem preenchidas com `fill_value` (ex.: genes do Pan ausentes no Mathys).
    fill_value : float, default=0.5
        Valor sentinela neutro atribuído às colunas faltantes.

    Returns
    -------
    sp.csr_matrix
        Matriz esparsa CSR com as colunas indicadas preenchidas com `fill_value`.
    """
    idx_lista: list[int] = [int(i) for i in np.asarray(indices_colunas).ravel()]
    if sp.issparse(matriz):
        matriz_lil = sp.lil_matrix(matriz, copy=True)
        if idx_lista:
            matriz_lil[:, idx_lista] = float(fill_value)
        return matriz_lil.tocsr()

    arr_mod = np.asarray(matriz, dtype=np.float32).copy()
    if idx_lista:
        arr_mod[:, idx_lista] = float(fill_value)
    return sp.csr_matrix(arr_mod)


def gerar_sweep_pan_com_faltantes_mathys(
    path_pan_h5ad: PathType,
    path_tracking_mathys_csv: PathType,
    dir_saida: PathType,
    path_orthbase_rds: PathType,
    n_componentes: int = 600,
    seed: int = 42,
    fill_value: float = 0.5,
    salvar_h5ad_modificado: bool = False,
    forcar_reprojecao: bool = False,
) -> NDArray[np.float32]:
    """Substitui no dataset Pan (~61k genes) as colunas faltantes no Mathys por 0.5 e projeta via rSWeeP.

    Reutiliza automaticamente a matriz projetada em disco caso já tenha sido gerada,
    a menos que `forcar_reprojecao=True`.

    Parameters
    ----------
    path_pan_h5ad : PathType
        Caminho para o arquivo `pan_anotado.h5ad` (58.326 células × 61.541 genes).
    path_tracking_mathys_csv : PathType
        Caminho para `tracking_genes_adicionados_mathys.csv` contendo `posicao_coluna`.
    dir_saida : PathType
        Diretório de saída onde serão gravados o `.mtx`, `.txt` e `.npy`.
    path_orthbase_rds : PathType
        Caminho para a base ortonormal congelada `orthbase_mproj_600d.rds` (61.541 × 600).
    n_componentes : int, default=600
        Número de dimensões da projeção SWeeP.
    seed : int, default=42
        Semente da base ortonormal rSWeeP.
    fill_value : float, default=0.5
        Valor sentinela atribuído às colunas faltantes.
    salvar_h5ad_modificado : bool, default=False
        Se True, persiste também o arquivo `pan_com_sentinela_05.h5ad`.
    forcar_reprojecao : bool, default=False
        Se True, força a substituição esparsa no `.h5ad` e reexecução do `ProjetorSWeePR`.

    Returns
    -------
    NDArray[np.float32]
        Matriz SWeeP projetada (58.326 × 600) em `float32`.
    """
    dir_out = Path(dir_saida)
    dir_out.mkdir(parents=True, exist_ok=True)

    path_txt = dir_out / "matriz_sweep_pan_faltantes_mathys.txt"
    path_npy = dir_out / "matriz_sweep_pan_faltantes_mathys.npy"
    path_mtx = dir_out / "matrix_pan_faltantes_mathys.mtx"
    path_h5ad_out = dir_out / "pan_com_sentinela_mathys_05.h5ad"

    if not forcar_reprojecao:
        if path_npy.exists():
            print(f"[Teste 1 Pan->Pan Faltantes] Carregando NPY em cache: {path_npy}")
            return carregar_matriz_sweep_600d(path_npy)
        if path_txt.exists():
            print(f"[Teste 1 Pan->Pan Faltantes] Carregando TXT em cache: {path_txt}")
            arr = carregar_matriz_sweep_600d(path_txt)
            np.save(path_npy, arr)
            return arr

    if not Path(path_pan_h5ad).exists():
        raise FileNotFoundError(
            f"Dataset Pan original não encontrado em: {path_pan_h5ad}"
        )
    if not Path(path_tracking_mathys_csv).exists():
        raise FileNotFoundError(
            f"Arquivo de tracking das colunas faltantes do Mathys não encontrado em: {path_tracking_mathys_csv}"
        )

    df_tracking = pl.read_csv(str(path_tracking_mathys_csv))
    indices_faltantes = df_tracking["posicao_coluna"].to_list()
    print(
        f"[Teste 1 Pan->Pan Faltantes] Substituindo {len(indices_faltantes)} colunas "
        f"faltantes do Mathys por {fill_value} no dataset Pan ({path_pan_h5ad})..."
    )

    adata_pan = ad.read_h5ad(str(path_pan_h5ad))
    assert adata_pan.X is not None
    x_mod = substituir_colunas_faltantes_esparso(
        matriz=adata_pan.X,
        indices_colunas=indices_faltantes,
        fill_value=fill_value,
    )

    if salvar_h5ad_modificado:
        adata_pan.X = x_mod
        adata_pan.write_h5ad(str(path_h5ad_out), compression="gzip")

    print(f"[Teste 1 Pan->Pan Faltantes] Exportando Matrix Market (.mtx): {path_mtx}")
    sio.mmwrite(str(path_mtx), x_mod)

    del adata_pan, x_mod
    gc.collect()

    ProjetorSWeePR.verificar_e_instalar_dependencias_r()
    projetor = ProjetorSWeePR(
        path_matriz=str(path_mtx),
        path_saida=str(path_txt),
        n_componentes=n_componentes,
        seed=seed,
        path_orthbase=str(path_orthbase_rds),
    )
    projetor.projetar()

    if projetor.Wswp is None:
        raise RuntimeError(
            "ProjetorSWeePR retornou matriz nula na projeção do Pan com faltantes do Mathys."
        )
    wswp_arr = np.asarray(projetor.Wswp, dtype=np.float32)
    if np.isnan(wswp_arr).any():
        raise RuntimeError(
            "Valores NaN detectados na projeção SWeeP do Pan com faltantes do Mathys."
        )

    np.save(path_npy, wswp_arr)
    return wswp_arr


def carregar_matriz_sweep_600d(
    path_arquivo: PathType,
    tem_cabecalho: bool | None = None,
    separador: str | None = None,
) -> NDArray[np.float32]:
    r"""Carrega uma matriz SWeeP 600D a partir de arquivo `.npy`, `.txt` ou `.csv` em `float32`.

    Parameters
    ----------
    path_arquivo : PathType
        Caminho do arquivo `.npy`, `.txt` ou `.csv`.
    tem_cabecalho : bool | None, optional
        Se None, detecta automaticamente se a primeira linha contém cabeçalho textual (ex.: `V1`).
    separador : str | None, optional
        Se None, detecta automaticamente entre tabulação (`\\t`), vírgula (`,`) ou espaço.

    Returns
    -------
    NDArray[np.float32]
        Matriz bidimensional em `float32`.
    """
    caminho = Path(path_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo SWeeP não encontrado: {caminho}")

    if caminho.suffix.lower() == ".npy":
        arr = np.asarray(np.load(caminho), dtype=np.float32)
    else:
        with open(caminho, encoding="utf-8") as f:
            primeira_linha = f.readline().strip()

        if separador is None:
            if "\t" in primeira_linha:
                separador = "\t"
            elif "," in primeira_linha:
                separador = ","
            else:
                separador = " "

        if tem_cabecalho is None:
            primeiro_token = (
                primeira_linha.split(separador)[0].strip().strip('"').strip("'")
            )
            try:
                float(primeiro_token)
                tem_cabecalho = False
            except ValueError:
                tem_cabecalho = True

        df = pl.read_csv(
            str(caminho),
            separator=separador,
            has_header=tem_cabecalho,
        )
        arr = df.to_numpy().astype(np.float32, copy=False)

    if arr.ndim != 2:
        raise ValueError(
            f"A matriz SWeeP carregada de {caminho} deve ser 2D, mas possui shape {arr.shape}."
        )
    if np.isnan(arr).any():
        raise ValueError(f"A matriz SWeeP carregada de {caminho} contém valores NaN.")
    return arr


def calcular_metricas_geometricas_600d(
    sweep_referencia: NDArray[Any],
    sweep_avaliado: NDArray[Any],
    rotulos: Sequence[int] | NDArray[Any] | None = None,
    classes_validas: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    nomes_classes: Mapping[int, str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calcula métricas geométricas de reconstrução 600D globais e estratificadas por classe.

    Parameters
    ----------
    sweep_referencia : NDArray
        Matriz SWeeP de referência sem valores faltantes (N × D).
    sweep_avaliado : NDArray
        Matriz SWeeP avaliada — Antes (com sentinela 0.5) ou Pós-Hopfield (N × D).
    rotulos : Sequence[int] | NDArray | None, optional
        Vetor de rótulos celulares (N,). Se fornecido, gera tabela por classe em `classes_validas`.
    classes_validas : Sequence[int], default=(1, 2, 3, 4, 5, 6, 7)
        Rótulos canônicos avaliados no detalhamento por classe.
    nomes_classes : Mapping[int, str] | None, optional
        Mapeamento opcional de índice de classe para nome legível.

    Returns
    -------
    tuple[dict[str, float], pd.DataFrame]
        Dicionário com resumo global (`mse`, `rmse`, `mae`, `cosseno_medio`, `cosseno_mediano`,
        `pearson_medio`, `pearson_mediano`) e DataFrame estratificado por classe.
    """
    ref = np.asarray(sweep_referencia, dtype=np.float64)
    aval = np.asarray(sweep_avaliado, dtype=np.float64)
    if ref.shape != aval.shape:
        raise ValueError(
            f"Dimensões incompatíveis entre referência {ref.shape} e avaliado {aval.shape}."
        )

    diff = aval - ref
    mse_por_celula = np.mean(diff * diff, axis=1)
    mae_por_celula = np.mean(np.abs(diff), axis=1)

    norm_ref = np.linalg.norm(ref, axis=1)
    norm_aval = np.linalg.norm(aval, axis=1)
    denom_cos = np.maximum(norm_ref * norm_aval, 1e-12)
    cosseno_por_celula = np.sum(ref * aval, axis=1) / denom_cos

    ref_cent = ref - np.mean(ref, axis=1, keepdims=True)
    aval_cent = aval - np.mean(aval, axis=1, keepdims=True)
    denom_pearson = np.maximum(
        np.linalg.norm(ref_cent, axis=1) * np.linalg.norm(aval_cent, axis=1),
        1e-12,
    )
    pearson_por_celula = np.sum(ref_cent * aval_cent, axis=1) / denom_pearson

    resumo_global: dict[str, float] = {
        "mse": float(np.mean(mse_por_celula)),
        "rmse": float(np.sqrt(np.mean(mse_por_celula))),
        "mae": float(np.mean(mae_por_celula)),
        "cosseno_medio": float(np.mean(cosseno_por_celula)),
        "cosseno_mediano": float(np.median(cosseno_por_celula)),
        "cosseno_p10": float(np.percentile(cosseno_por_celula, 10.0)),
        "cosseno_p90": float(np.percentile(cosseno_por_celula, 90.0)),
        "pearson_medio": float(np.mean(pearson_por_celula)),
        "pearson_mediano": float(np.median(pearson_por_celula)),
    }

    mapa_nomes = nomes_classes if nomes_classes is not None else NOMES_CLASSES_PADRAO
    linhas_classes: list[dict[str, Any]] = []
    if rotulos is not None:
        y_arr = np.asarray(rotulos).ravel()
        for c in classes_validas:
            mask_c = y_arr == c
            if not np.any(mask_c):
                continue
            linhas_classes.append(
                {
                    "classe_id": int(c),
                    "tipo_celular": mapa_nomes.get(int(c), f"Classe {c}"),
                    "n_celulas": int(np.sum(mask_c)),
                    "mse": float(np.mean(mse_por_celula[mask_c])),
                    "mae": float(np.mean(mae_por_celula[mask_c])),
                    "cosseno_medio": float(np.mean(cosseno_por_celula[mask_c])),
                    "cosseno_mediano": float(np.median(cosseno_por_celula[mask_c])),
                    "pearson_medio": float(np.mean(pearson_por_celula[mask_c])),
                }
            )

    df_classes = pd.DataFrame(linhas_classes)
    return resumo_global, df_classes


def calibrar_hopfield_sweep_completo(
    sweep_referencia: NDArray[Any],
    sweep_faltante: NDArray[Any],
    rotulos_referencia: Sequence[int] | NDArray[Any] | None = None,
    betas: Sequence[float] = (15.0, 30.0, 50.0, 100.0, 200.0, 500.0),
    n_iters_list: Sequence[int] = (1, 2, 3),
    batch_size: int = 1024,
    mask_self: bool = True,
    modelo_mlp: Any | None = None,
    scaler_mlp: Any | None = None,
    classes_validas: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    device: str | torch.device | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Executa Grid Search de `beta` e `n_iters` na Rede Hopfield com memória completa do Pan.

    Parameters
    ----------
    sweep_referencia : NDArray
        Matriz SWeeP completa do conjunto Pan (58.326 × 600).
    sweep_faltante : NDArray
        Matriz SWeeP do conjunto Pan com colunas faltantes do Mathys preenchidas com 0.5.
    rotulos_referencia : Sequence[int] | NDArray | None, optional
        Rótulos celulares do Pan para avaliação estratificada e/ou classificador MLP.
    betas : Sequence[float], default=(15.0, 30.0, 50.0, 100.0, 200.0, 500.0)
        Valores candidatos de temperatura inversa `beta`.
    n_iters_list : Sequence[int], default=(1, 2, 3)
        Valores candidatos de iterações síncronas `n_iters`.
    batch_size : int, default=1024
        Tamanho do lote OOM-Safe para recuperação.
    mask_self : bool, default=True
        Se True, aplica máscara Leave-One-Out (i != j) para evitar auto-casamento trivial.
    modelo_mlp : Any | None, optional
        Modelo Keras `clasificador_pan.keras` opcional para medir ganho de acurácia em cada ponto do grid.
    scaler_mlp : Any | None, optional
        Scaler `scaler_celular_pan.pkl` correspondente ao `modelo_mlp`.
    classes_validas : Sequence[int], default=(1, 2, 3, 4, 5, 6, 7)
        Classes canônicas avaliadas no MLP.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        DataFrame ordenado com todos os experimentos do Grid Search e dicionário da melhor configuração.
    """
    ref_f32 = np.asarray(sweep_referencia, dtype=np.float32)
    falt_f32 = np.asarray(sweep_faltante, dtype=np.float32)

    metricas_antes, _ = calcular_metricas_geometricas_600d(
        sweep_referencia=ref_f32,
        sweep_avaliado=falt_f32,
        rotulos=rotulos_referencia,
        classes_validas=classes_validas,
    )

    acuracia_antes: float = float("nan")
    f1_macro_antes: float = float("nan")
    if (
        modelo_mlp is not None
        and scaler_mlp is not None
        and rotulos_referencia is not None
    ):
        aval_antes = avaliar_no_classificador_pan(
            sweep_600d=falt_f32,
            rotulos=rotulos_referencia,
            modelo_keras=modelo_mlp,
            scaler=scaler_mlp,
            classes_validas=classes_validas,
            extrair_latente_32d=False,
        )
        acuracia_antes = float(aval_antes["acuracia"])
        f1_macro_antes = float(aval_antes["f1_macro"])

    rede = ModernHopfieldNetwork(
        beta=float(betas[0]),
        n_iters=int(n_iters_list[0]),
        binary=False,
        normalize=True,
        device=device,
    )
    rede.store(ref_f32)

    registros: list[dict[str, float]] = []
    for beta in betas:
        for n_it in n_iters_list:
            rede.beta = float(beta)
            rede.n_iters = int(n_it)

            reconstruido = rede.retrieve(
                falt_f32,
                batch_size=batch_size,
                mask_self=mask_self,
            )
            met_pos, _ = calcular_metricas_geometricas_600d(
                sweep_referencia=ref_f32,
                sweep_avaliado=reconstruido,
                rotulos=rotulos_referencia,
                classes_validas=classes_validas,
            )

            acc_pos = float("nan")
            f1_pos = float("nan")
            if (
                modelo_mlp is not None
                and scaler_mlp is not None
                and rotulos_referencia is not None
            ):
                aval_pos = avaliar_no_classificador_pan(
                    sweep_600d=reconstruido,
                    rotulos=rotulos_referencia,
                    modelo_keras=modelo_mlp,
                    scaler=scaler_mlp,
                    classes_validas=classes_validas,
                    extrair_latente_32d=False,
                )
                acc_pos = float(aval_pos["acuracia"])
                f1_pos = float(aval_pos["f1_macro"])

            diag = rede.ultimo_diagnostico_atencao
            registros.append(
                {
                    "beta": float(beta),
                    "n_iters": float(n_it),
                    "mse_antes": float(metricas_antes["mse"]),
                    "mse": float(met_pos["mse"]),
                    "mae_antes": float(metricas_antes["mae"]),
                    "mae": float(met_pos["mae"]),
                    "cosseno_antes": float(metricas_antes["cosseno_medio"]),
                    "cosseno_medio": float(met_pos["cosseno_medio"]),
                    "pearson_antes": float(metricas_antes["pearson_medio"]),
                    "pearson_medio": float(met_pos["pearson_medio"]),
                    "acuracia_antes": acuracia_antes,
                    "acuracia_mlp": acc_pos,
                    "f1_macro_antes": f1_macro_antes,
                    "f1_macro_mlp": f1_pos,
                    "max_weight_mean": float(diag.get("max_weight_mean", 0.0)),
                    "effective_neighbors_mean": float(
                        diag.get("effective_neighbors_mean", 0.0)
                    ),
                }
            )

    df_grid = pd.DataFrame(registros)
    if (
        modelo_mlp is not None
        and scaler_mlp is not None
        and rotulos_referencia is not None
    ):
        df_grid = df_grid.sort_values(
            by=["acuracia_mlp", "cosseno_medio", "mse"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    else:
        df_grid = df_grid.sort_values(
            by=["cosseno_medio", "mse"],
            ascending=[False, True],
        ).reset_index(drop=True)

    melhor_cfg: dict[str, float] = {
        str(k): float(v) for k, v in df_grid.iloc[0].to_dict().items()
    }
    return df_grid, melhor_cfg


def avaliar_no_classificador_pan(
    sweep_600d: NDArray[Any],
    rotulos: Sequence[int] | NDArray[Any],
    modelo_keras: Any,
    scaler: Any,
    classes_validas: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    nomes_classes: Sequence[str] | None = None,
    extrair_latente_32d: bool = True,
) -> dict[str, Any]:
    """Avalia uma matriz SWeeP 600D no classificador celular pré-treinado do conjunto Pan.

    Parameters
    ----------
    sweep_600d : NDArray
        Matriz SWeeP 600D a ser classificada (N × 600).
    rotulos : Sequence[int] | NDArray
        Vetor de rótulos celulares reais (N,). Apenas células pertencentes a `classes_validas`
        são avaliadas.
    modelo_keras : Any
        Modelo Keras carregado de `clasificador_pan.keras`.
    scaler : Any
        Scaler carregado de `scaler_celular_pan.pkl`.
    classes_validas : Sequence[int], default=(1, 2, 3, 4, 5, 6, 7)
        Rótulos canônicos válidos (1 a 7).
    nomes_classes : Sequence[str] | None, optional
        Nomes legíveis das classes para o relatório.
    extrair_latente_32d : bool, default=True
        Se True, extrai também as ativações da penúltima camada densa (32 neurônios)
        para visualização PCA/UMAP.

    Returns
    -------
    dict[str, Any]
        Dicionário contendo `acuracia`, `f1_macro`, `f1_weighted`, `y_real`, `y_pred`,
        `probabilidades`, `matriz_confusao`, `df_relatorio_classes` e `latente_32d`.
    """
    x_arr = np.asarray(sweep_600d, dtype=np.float32)
    y_arr = np.asarray(rotulos).ravel()

    mascara = np.isin(y_arr, list(classes_validas))
    x_filtrado = x_arr[mascara]
    y_real = y_arr[mascara].astype(np.int32)

    x_norm = np.asarray(scaler.transform(x_filtrado), dtype=np.float32)
    probabilidades = np.asarray(
        modelo_keras.predict(x_norm, verbose=0), dtype=np.float32
    )
    y_pred = (np.argmax(probabilidades, axis=1) + 1).astype(np.int32)

    acuracia = float(accuracy_score(y_real, y_pred))
    f1_macro = float(
        f1_score(
            y_real,
            y_pred,
            labels=list(classes_validas),
            average="macro",
            zero_division=0,
        )
    )
    f1_weighted = float(
        f1_score(
            y_real,
            y_pred,
            labels=list(classes_validas),
            average="weighted",
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_real, y_pred, labels=list(classes_validas))

    prec, rec, f1_c, sup = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=list(classes_validas),
        zero_division=0,
    )
    rotulos_nomes = (
        list(nomes_classes)
        if nomes_classes is not None
        else [NOMES_CLASSES_PADRAO.get(int(c), f"Classe {c}") for c in classes_validas]
    )
    df_relatorio = pd.DataFrame(
        {
            "classe_id": [int(c) for c in classes_validas],
            "tipo_celular": rotulos_nomes,
            "precision": np.asarray(prec, dtype=np.float64),
            "recall": np.asarray(rec, dtype=np.float64),
            "f1_score": np.asarray(f1_c, dtype=np.float64),
            "support": np.asarray(sup, dtype=np.int64),
        }
    )

    latente_32d: NDArray[np.float32] | None = None
    if (
        extrair_latente_32d
        and hasattr(modelo_keras, "layers")
        and len(modelo_keras.layers) >= 2
    ):
        try:
            import tensorflow as tf  # type: ignore[import-untyped]

            penultima_camada = modelo_keras.layers[-2]
            extrator = tf.keras.Model(
                inputs=modelo_keras.inputs,
                outputs=penultima_camada.output,
            )
            latente_32d = np.asarray(
                extrator.predict(x_norm, verbose=0), dtype=np.float32
            )
        except Exception as exc:
            logger.warning(
                "Falha ao extrair espaço latente 32D da penúltima camada do modelo Keras: %s",
                exc,
                exc_info=True,
            )
            latente_32d = None

    return {
        "acuracia": acuracia,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "y_real": y_real,
        "y_pred": y_pred,
        "probabilidades": probabilidades,
        "matriz_confusao": cm,
        "df_relatorio_classes": df_relatorio,
        "latente_32d": latente_32d,
    }
