from os import path

import recommender_system_leads

base_path = path.dirname(path.dirname(recommender_system_leads.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'models')
