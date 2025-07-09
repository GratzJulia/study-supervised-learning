import pandas as pd
import numpy as np

def generateDataset():
    np.random.seed(42)
    num_samples = 3333

    # account_length: Int, min/max, próximo de uma distribuição uniforme ou normal truncada
    account_length = np.random.randint(1, 244, num_samples) # Min 1, Max 243

    # total_day_charge: Float, aproximar com distribuição normal truncada ou ajuste de range
    # Para simplificar, usaremos uma normal truncada ou clip para garantir min/max
    total_day_charge = np.random.normal(loc=30.562, scale=9.259, size=num_samples)
    total_day_charge = np.clip(total_day_charge, 0.000, 59.640) # Garante min e max

    # total_eve_charge: Float
    total_eve_charge = np.random.normal(loc=17.084, scale=4.311, size=num_samples)
    total_eve_charge = np.clip(total_eve_charge, 0.000, 30.910)

    # total_night_charge: Float
    total_night_charge = np.random.normal(loc=9.039, scale=2.276, size=num_samples)
    total_night_charge = np.clip(total_night_charge, 1.040, 17.770)

    # total_intl_charge: Float
    total_intl_charge = np.random.normal(loc=2.765, scale=0.754, size=num_samples)
    total_intl_charge = np.clip(total_intl_charge, 0.000, 5.400)

    # customer_service_calls: Int, min/max
    # Usar random.randint e clip para forçar a distribuição de inteiros
    customer_service_calls = np.random.randint(0, 10, num_samples) # Min 0, Max 9

    # churn: Binário (0 ou 1) com base na média (proporção de 1s)
    churn_rate = 0.145
    churn = np.random.choice([0, 1], size=num_samples, p=[1 - churn_rate, churn_rate])

    churn_df = pd.DataFrame({
        'account_length': account_length,
        'total_day_charge': total_day_charge,
        'total_eve_charge': total_eve_charge,
        'total_night_charge': total_night_charge,
        'total_intl_charge': total_intl_charge,
        'customer_service_calls': customer_service_calls,
        'churn': churn
    })

    # Ajustar tipos de dados onde necessário
    churn_df['account_length'] = churn_df['account_length'].astype(int)
    churn_df['customer_service_calls'] = churn_df['customer_service_calls'].astype(int)
    churn_df['churn'] = churn_df['churn'].astype(int)

    # Para `total_day_charge`, `total_eve_charge`, `total_night_charge`, `total_intl_charge` que são float64, numpy.random.normal já os gera como float. Podemos arredondar para um número razoável de casas decimais para simular os dados originais
    churn_df['total_day_charge'] = churn_df['total_day_charge'].round(3)
    churn_df['total_eve_charge'] = churn_df['total_eve_charge'].round(3)
    churn_df['total_night_charge'] = churn_df['total_night_charge'].round(3)
    churn_df['total_intl_charge'] = churn_df['total_intl_charge'].round(3)

    print(churn_df.head())
    print(churn_df.info())
    print(churn_df.describe())

    return churn_df