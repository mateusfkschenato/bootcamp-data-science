# Remove das pastas oficiais os arquivos da rodada da Random Forest (6 features) que não são mais gerados.
# Antes de rodar: as cópias desses arquivos já estão em <etapa>\historico_rf6\ (confira).
# Os de otimizacao_minutos\ são duplicatas do que agora está em feature_selection\ e selecao_algoritmos\
# (os scripts foram para otimizacao_minutos\experimentos\, como registro).
#
# Uso, na raiz do repositório (C:\projetos\bootcamp-data-science), no PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\limpar_arquivos_rodada_rf6.ps1
# Como o projeto está no git, tudo o que sair aqui continua recuperável pelo histórico.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$arquivos = @(
    "feature_selection\curva_n_features.csv",
    "feature_selection\votos_feature_selection.csv",
    "feature_selection\features_guia_modelagem_avaliacao.csv",
    "feature_selection\graficos\05_curva_desempenho_n_features.png",
    "feature_selection\graficos\02_caminho_regularizacao.png",
    "feature_selection\graficos\04_rfecv.png",
    "feature_selection\graficos\03_importancia_e_permutacao.png",
    "feature_selection\graficos\06_mudanca_de_regime.png",
    "feature_selection\graficos\01_filtro_correlacao_info_mutua.png",
    "selecao_algoritmos\limiares_validacao.csv",
    "selecao_algoritmos\modelo_final_planoB.joblib",
    "selecao_algoritmos\previsoes_validacao_teste.csv",
    "selecao_algoritmos\modelo_final_planoA.joblib",
    "selecao_algoritmos\graficos\08_arvore_rasa_fluxograma.png",
    "selecao_algoritmos\graficos\03_cv_vs_validacao.png",
    "selecao_algoritmos\graficos\07_importancia_modelo_escolhido.png",
    "selecao_algoritmos\graficos\01_baseline_vs_algoritmos.png",
    "selecao_algoritmos\graficos\06_matriz_confusao_teste.png",
    "selecao_algoritmos\graficos\02_pr_auc_por_prova.png",
    "selecao_algoritmos\graficos\04_escada_de_complexidade.png",
    "selecao_algoritmos\graficos\05_limiar_recall_precisao.png",
    "analise_modelos\metricas_cv_temporal.csv",
    "analise_modelos\modelo_para_uso_no_teste.csv",
    "analise_modelos\calibracao_aplicada.csv",
    "analise_modelos\custo_no_teste.csv",
    "analise_modelos\comparacao_pareada_teste.csv",
    "analise_modelos\modelo_para_uso_limiares.csv",
    "analise_modelos\comparacao_pareada_validacao.csv",
    "analise_modelos\tabela_unica_validacao.csv",
    "analise_modelos\baselines_validacao.csv",
    "analise_modelos\custo_conforme_k_validacao.csv",
    "analise_modelos\regras_para_o_patio.csv",
    "analise_modelos\modelo_para_uso_contra_regra.csv",
    "analise_modelos\dependencia_parcial_sentido.csv",
    "analise_modelos\metricas_validacao.csv",
    "analise_modelos\graficos\03_custo_conforme_k.png",
    "analise_modelos\graficos\09_contribuicoes_global_e_local.png",
    "analise_modelos\graficos\01_comparacao_cv_validacao.png",
    "analise_modelos\graficos\08_desempenho_por_segmento.png",
    "analise_modelos\graficos\07_curva_de_aprendizado.png",
    "analise_modelos\graficos\02_curvas_roc_e_precisao_recall.png",
    "analise_modelos\graficos\06_a_cada_100_cestoes_no_teste.png",
    "analise_modelos\graficos\10_dependencia_parcial.png",
    "verificacao_validacao\contagens_por_etapa.csv",
    "verificacao_validacao\sensibilidade_historico.csv",
    "verificacao_validacao\gap_treino_cv_validacao_teste.csv",
    "verificacao_validacao\graficos\03_tipos_de_validacao_cruzada.png",
    "verificacao_validacao\graficos\04_drift_das_features.png",
    "verificacao_validacao\graficos\08_validacao_de_negocio.png",
    "verificacao_validacao\graficos\02_complexidade_subajuste_sobreajuste.png",
    "verificacao_validacao\graficos\05_relacao_feature_alvo_por_particao.png",
    "verificacao_validacao\graficos\07_walk_forward_mensal.png",
    "verificacao_validacao\graficos\06_monitoramento_simulado.png",
    "verificacao_validacao\graficos\01_gap_treino_cv_validacao_teste.png",
    "interpretacao_modelos\graficos\05_lime_estabilidade.png",
    "interpretacao_modelos\graficos\03_shap_visao_geral.png",
    "interpretacao_modelos\graficos\02_pdp_ice_e_casos_impossiveis.png",
    "interpretacao_modelos\graficos\01_importancia_global_tres_reguas.png",
    "interpretacao_modelos\graficos\04_shap_dois_cestoes.png",
    "otimizacao_minutos\modelo_minutos.py",
    "otimizacao_minutos\features_minutos.py",
    "otimizacao_minutos\dataset_minutos.csv",
    "otimizacao_minutos\modelo_minutos.joblib",
    "otimizacao_minutos\features_finais.json"
)
$faltando = @()
foreach ($a in $arquivos) {
    $nome = Split-Path $a -Leaf
    $etapa = ($a -split "\\")[0]
    $copia1 = Join-Path $etapa (Join-Path "historico_rf6" $nome)
    $copia2 = Join-Path $etapa (Join-Path "historico_rf6\graficos" $nome)
    if ($etapa -ne "otimizacao_minutos" -and -not ((Test-Path $copia1) -or (Test-Path $copia2))) { $faltando += $a }
}
if ($faltando.Count -gt 0) {
    Write-Host "Sem cópia em historico_rf6, nada foi apagado:" -ForegroundColor Red
    $faltando | ForEach-Object { Write-Host "  $_" }
    exit 1
}
$apagados = 0
foreach ($a in $arquivos) {
    if (Test-Path $a) { Remove-Item $a; Write-Host "apagado: $a"; $apagados++ }
}
Write-Host "$apagados arquivos apagados." -ForegroundColor Green
Remove-Item $PSCommandPath
Write-Host "Este script também foi apagado (não é mais necessário)."
