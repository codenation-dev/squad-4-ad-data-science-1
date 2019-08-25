# Runs Recommender Scripts
import Class_Recommender_systems_1
import Class_Recommender_systems_2
import Class_Recommender_systems_3

# Get files:
import pandas as pd
market = pd.read_csv("market.csv")
rec_port1_diego = pd.read_csv("rec_port1_diego.csv")
rec_port2_diego = pd.read_csv("rec_port2_diego.csv")
rec_port3_diego = pd.read_csv("rec_port3_diego.csv")

## Ranking
# Concat files:
rec_frames = [rec_port1_diego, rec_port2_diego,rec_port3_diego]
rec_portfolios = pd.concat(rec_frames)

# Count:
cross_rec = rec_portfolios['id'].value_counts()
df_cross_rec = cross_rec.rename_axis('id').reset_index(name='Recomendacoes')


# Complete Recommender data:
df_cross_rec = pd.merge(df_cross_rec, market, on='id', how='left')

# Output:
df_cross_rec.head(50).to_csv('final_recommendation.csv', index=False)