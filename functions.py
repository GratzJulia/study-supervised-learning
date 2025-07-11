import pandas as pd
import numpy as np

def generate_classification_dataset():
    generator = np.random.default_rng(42)
    num_samples = 3333

    # account_length: Int, Min 1/Max 243, próximo de uma distribuição uniforme ou normal truncada
    account_length = generator.integers(1, 244, num_samples)

    # total_day_charge: Float, aproximar com distribuição normal truncada
    total_day_charge = generator.normal(loc=30.562, scale=9.259, size=num_samples)
    total_day_charge = np.clip(total_day_charge, 0.000, 59.640).round(2) # Garante min e max

    # total_eve_charge: Float
    total_eve_charge = generator.normal(loc=17.084, scale=4.311, size=num_samples)
    total_eve_charge = np.clip(total_eve_charge, 0.000, 30.910).round(2)

    # total_night_charge: Float
    total_night_charge = generator.normal(loc=9.039, scale=2.276, size=num_samples)
    total_night_charge = np.clip(total_night_charge, 1.040, 17.770).round(2)

    # total_intl_charge: Float
    total_intl_charge = generator.normal(loc=2.765, scale=0.754, size=num_samples)
    total_intl_charge = np.clip(total_intl_charge, 0.000, 5.400).round(2)

    # customer_service_calls: Int, min/max
    # Usar random.randint e clip para forçar a distribuição de inteiros
    customer_service_calls = generator.integers(0, 10, num_samples) # Min 0, Max 9

    # churn: Binário (0 ou 1) com base na média (proporção de 1s)
    churn_rate = 0.145
    churn = generator.choice([0, 1], size=num_samples, p=[1 - churn_rate, churn_rate])

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

    # print(churn_df.head())
    # print(churn_df.info())
    # print(churn_df.describe())

    return churn_df

def generate_regression_dataset():
    generator = np.random.default_rng(42) 
    num_samples = 4546

    # tv
    tv = generator.normal(loc=54062.912, scale=26104.942, size=num_samples)
    tv = np.clip(tv, 10000.000, 100000.000).round(2)

    # radio
    radio = generator.normal(loc=18157.533, scale=9663.260, size=num_samples)
    radio = np.clip(radio, 0.680, 48871.160).round(2)

    # social_media
    social_media = generator.normal(loc=3323.473, scale=2211.254, size=num_samples)
    social_media = np.clip(social_media, 0.030, 13981.660).round(2)

    sales = generator.normal(loc=192413.332, scale=93019.873, size=num_samples)
    sales = np.clip(sales, 31199.410, 364079.750).round(2)

    sales_df = pd.DataFrame({
        'tv': tv,
        'radio': radio,
        'social_media': social_media,
        'sales': sales
    })

    # print(sales_df.head())
    # print(sales_df.info())
    # print(sales_df.describe())

    return sales_df