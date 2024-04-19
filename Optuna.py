### Optuna ###
import optuna

# Define objective function (during optimzation Optuna repeats this function)
class Objective:
    def __init__(self, min_x, max_x, depth_range, split_range, leaf_range,n_estimators_range, learning_rate_range):
        self.min_x = min_x
        self.max_x = max_x
        self.depth_range = depth_range
        self.split_range = split_range
        self.leaf_range = leaf_range
        self.n_estimators_range = n_estimators_range
        self.learning_rate_range = learning_rate_range

    def __call__(self, trial):
        x = trial.suggest_float("x", self.min_x, self.max_x)
        max_depth = trial.suggest_int("max_depth", *self.depth_range)
        min_samples_split = trial.suggest_int("min_samples_split", *self.split_range)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", *self.leaf_range)
        n_estimators = trial.suggest_int("n_estimators", *self.n_estimators_range)
        learning_rate = trial.suggest_float("learning_rate", *self.learning_rate_range)
        
        objective_value = (x - 2) ** 2 + max_depth + min_samples_split + min_samples_leaf
                
        return objective_value #Returns the evaluation score of the function

# Execute an optimization by using an `Objective` instance.
n_estimators_range = (100,1000) # Range for n_estimators
depth_range = (10, 50)  # Range for max_depth
split_range = (2, 32)   # Range for min_samples_split
leaf_range = (1, 32)    # Range for min_samples_leaf
learning_rate_range = (0.01,0.1) # Range for learning_rate

''' Note: num_leaves = 2^(max_depth) However, considering that in LGBM a leaf-wise tree is 
deeper than a level-wise tree you need to be careful about overfitting! 
As a result, It is necessary to tune num_leaves with the max_depth together.
'''

# Instantiate a study object. A study corresponds to an optimization task, i.e., a set of trials.
study = optuna.create_study() 
n_trials = 200 # The number of iterations, default = 100 
objective_instance = Objective(-100, 100,depth_range, split_range, leaf_range,n_estimators_range,learning_rate_range)
study.optimize(objective_instance,n_jobs=2, n_trials=n_trials)

#Best parameters
best_params = study.best_params()
print("The best params with Optuna: ", n_trials, best_params)

### Visualizing Optuna results ###

#Plot optimization history of all trials in a study
fig_optuna = optuna.visualization.plot_optimization_history(study)
fig_optuna.show()

# Plot the high-dimensional parameter relationships in a study
# Note: if a parameter contains missing values, a trial with missing values is not plotted
fig_optuna_coord = optuna.visualization.plot_parallel_coordinate(study)
fig_optuna_coord.show()

fig_plot_contour = optuna.visualization.plot_contour(study, params = ['min_samples_split','max_depth'])
fig_plot_contour.show()

#Scatter plot objective value vs. hypermparameter
fig_slice = optuna.visualization.plot_slice(study, params = ['min_samples_split','max_depth','min_samples_leaf','learning_rate'])
fig_slice.show()

#Optuna Parameters importance
fig_importance = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration")
fig_importance.show()