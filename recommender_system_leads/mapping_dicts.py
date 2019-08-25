import numpy as np

# full_dict = {"de_faixa_faturamento_estimado_grupo":
#                             {"ATE R$ 81.000,00": 0,
#                             "DE R$ 81.000,01 A R$ 360.000,00": 1,
#                             "DE R$ 360.000,01 A R$ 1.500.000,00": 2,
#                             "DE R$ 1.500.000,01 A R$ 4.800.000,00": 3,
#                             "DE R$ 4.800.000,01 A R$ 10.000.000,00": 4,
#                             "DE R$ 10.000.000,01 A R$ 30.000.000,00": 5,
#                             "DE R$ 30.000.000,01 A R$ 100.000.000,00": 6,
#                             "DE R$ 100.000.000,01 A R$ 300.000.000,00": 7,
#                             "DE R$ 300.000.000,01 A R$ 500.000.000,00": 8,
#                             "DE R$ 500.000.000,01 A 1 BILHAO DE REAIS": 9,
#                             "ACIMA DE 1 BILHAO DE REAIS": 10},
#             "de_faixa_faturamento_estimado":
#                             {"ATE R$ 81.000,00": 0,
#                             "DE R$ 81.000,01 A R$ 360.000,00": 1,
#                             "DE R$ 360.000,01 A R$ 1.500.000,00": 2,
#                             "DE R$ 1.500.000,01 A R$ 4.800.000,00": 3,
#                             "DE R$ 4.800.000,01 A R$ 10.000.000,00": 4,
#                             "DE R$ 10.000.000,01 A R$ 30.000.000,00": 5,
#                             "DE R$ 30.000.000,01 A R$ 100.000.000,00": 6,
#                             "DE R$ 100.000.000,01 A R$ 300.000.000,00": 7,
#                             "DE R$ 300.000.000,01 A R$ 500.000.000,00": 8,
#                             "DE R$ 500.000.000,01 A 1 BILHAO DE REAIS": 9,
#                             "ACIMA DE 1 BILHAO DE REAIS": 10,
#                             "SEM INFORMACAO": np.nan,}}


COLUMNS_WITH_DUPLICATED_INFO = ["idade_emp_cat", "de_faixa_faturamento_estimado", "de_faixa_faturamento_estimado_grupo", "nm_divisao"]
COLUMNS_WITH_TREATED_INFO = ["dt_situacao"]

COLUMNS_WITH_ADDITIONAL_LABEL_FOR_NULL = ["de_saude_rescencia", "de_saude_tributaria", "de_nivel_atividade"]
COLUMNS_TO_BOOL = ["fl_passivel_iss", "fl_simples_irregular", "fl_veiculo", "fl_antt", "fl_spa", "fl_telefone", "fl_rm"]

MAP_TO_NUMERICAL_ENCODING = {
    "de_nivel_atividade" : {"MUITO BAIXA": 0,
                            "BAIXA": 1,
                            "MEDIA": 2,
                            "ALTA": 3,
                            np.nan: 4},
    "de_saude_tributaria" : {"VERDE": 0,
                            "AZUL": 1,
                            "AMARELO": 2,
                            "CINZA": 3,
                            "LARANJA": 4,
                            "VERMELHO": 5,
                            np.nan: 0},
    "de_saude_rescencia" : {"ACIMA DE 1 ANO": 5,
                        "ATE 1 ANO": 4,
                        "ATE 6 MESES": 3,
                        "ATE 3 MESES": 2,
                        "SEM INFORMACAO": 1,
                        np.nan : 1} ,
    "fl_rm":{"SIM": True,
             "NAO": False}
}



FILL_BINARY_N_CONTINUOUS = { "nu_meses_rescencia": {
                                "value": None,
                                "method": "mean"
                            },
                            "sg_uf_matriz": {
                                "value": "INDETERMINADA",
                                "method": "constant",
                            },
                            "dt_situacao": {
                                "value": None,
                                "method": "mode"
                            },
                            "fl_optante_simei": {
                                "value": False,
                                "method" : "constant"
                            },
                            "fl_optante_simples": {
                                "value": False,
                                "method": "constant"
                            },
                            "qt_socios": {
                                "value": None,
                                "method": "median",
                            },
                            "qt_socios_pf": {
                                "value": None,
                                "method": "median",
                            },
                            "qt_socios_pj": {
                                "value": None,
                                "method": "median",
                            },
                            "qt_socios_masculino": {
                                "value": None,
                                "method": "median",
                            },
                            "qt_socios": {
                                "value": None,
                                "method": "median",
                            },
                            "qt_socios_st_regular": {
                                "value": None,
                                "method": "median",
                            },
                            "idade_media_socios": {
                                "value": None,
                                "method": "median",
                            },
                            "idade_minima_socios": {
                                "value": None,
                                "method": "median",
                            },
                            "idade_maxima_socios": {
                                "value": None,
                                "method": "median",
                            },
                            "nm_meso_regiao": {
                                "value": "INDETERMINADA",
                                "method": "constant",
                            },
                            "nm_micro_regiao": {
                                "value": "INDETERMINADA",
                                "method": "constant",
                            },
                            "vl_faturamento_estimado_aux" : {
                                "value": None,
                                "method": "median",
                            },
                            "vl_faturamento_estimado_grupo_aux" : {
                                "value": None,
                                "method": "median",
                            },

                            }

NUMERICAL_COLUMNS_TO_SCALE = ["idade_empresa_anos", "vl_total_veiculos_antt", "vl_total_veiculos_leves",
                "vl_total_veiculos_pesados", "vl_total_veiculos_pesados_grupo", "vl_total_veiculos_leves_grupo",
                "vl_total_veiculos_antt_grupo", "nu_meses_rescencia", "empsetorcensitariofaixarendapopulacao", "qt_socios",
                "qt_socios_pf", "qt_socios_pj", "idade_media_socios", "idade_maxima_socios", "idade_minima_socios",
                "qt_socios_st_regular", "qt_socios_masculino", "qt_funcionarios_coligados", "qt_funcionarios_coligados_gp",
                "media_funcionarios_coligados_gp", "max_funcionarios_coligados_gp", "min_funcionarios_coligados_gp",
                "sum_faturamento_estimado_coligadas", "vl_faturamento_estimado_aux", "vl_faturamento_estimado_grupo_aux",
                "qt_ex_funcionarios", "qt_funcionarios_grupo", "qt_funcionarios", "qt_funcionarios_12meses",
                    "qt_funcionarios_24meses", "qt_filiais", "dt_situacao_numeric", "dt_situacao_year"]
