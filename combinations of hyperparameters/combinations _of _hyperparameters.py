# Define the parameter grid
param_grid = {
    'pretrained_model': ["all-MiniLM-L6-v2", "roberta-base", "bert-base-uncased"],
    'index_dimension': [256, 512, 1024],
    'similarity_threshold': [0.7, 0.8, 0.9]
}


param_combinations = list(ParameterGrid(param_grid))

# Train and evaluate model for each combination of hyperparameters
results = []
for params in param_combinations:
    result = train_and_evaluate(**params)
    results.append({**params, **result})


results_df = pd.DataFrame(results)

best_result = results_df.loc[results_df['precision'].idxmax()]
print("Best parameters found:")
print(best_result)
