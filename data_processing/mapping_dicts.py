import numpy as np

full_dict = {"de_faixa_faturamento_estimado_grupo": 
                            {"ATE R$ 81.000,00": 0,
                            "DE R$ 81.000,01 A R$ 360.000,00": 1,
                            "DE R$ 360.000,01 A R$ 1.500.000,00": 2,
                            "DE R$ 1.500.000,01 A R$ 4.800.000,00": 3,
                            "DE R$ 4.800.000,01 A R$ 10.000.000,00": 4,
                            "DE R$ 10.000.000,01 A R$ 30.000.000,00": 5,
                            "DE R$ 30.000.000,01 A R$ 100.000.000,00": 6,
                            "DE R$ 100.000.000,01 A R$ 300.000.000,00": 7,
                            "DE R$ 300.000.000,01 A R$ 500.000.000,00": 8,
                            "DE R$ 500.000.000,01 A 1 BILHAO DE REAIS": 9,
                            "ACIMA DE 1 BILHAO DE REAIS": 10},
            "de_faixa_faturamento_estimado":
                            {"ATE R$ 81.000,00": 0,
                            "DE R$ 81.000,01 A R$ 360.000,00": 1,
                            "DE R$ 360.000,01 A R$ 1.500.000,00": 2,
                            "DE R$ 1.500.000,01 A R$ 4.800.000,00": 3,
                            "DE R$ 4.800.000,01 A R$ 10.000.000,00": 4,
                            "DE R$ 10.000.000,01 A R$ 30.000.000,00": 5,
                            "DE R$ 30.000.000,01 A R$ 100.000.000,00": 6,
                            "DE R$ 100.000.000,01 A R$ 300.000.000,00": 7,
                            "DE R$ 300.000.000,01 A R$ 500.000.000,00": 8,
                            "DE R$ 500.000.000,01 A 1 BILHAO DE REAIS": 9,
                            "ACIMA DE 1 BILHAO DE REAIS": 10,
                            "SEM INFORMACAO": np.nan,}} 