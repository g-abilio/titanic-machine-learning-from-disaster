# Adding modules
using Pkg
Pkg.add("DecisionTree")
Pkg.add("ScikitLearn")

# Adding custom modules
include("train_df_creation.jl")
include("test_df_creation.jl")

# Import modules
using DecisionTree
using ScikitLearn
using ScikitLearn.CrossValidation

# Function that will be responsible to make the
# model instantiation, learning and accuracy testing,
# as well as train-test split

function model_development()

    # Picking up df_train

    df_train = df_train_creation()

    # Train-test split

    y = df_train.Survived
    x = Matrix(select(df_train, Not("Survived")))

    # Model instantiation

    global model = RandomForestClassifier(n_trees=1000)

    # Model fitting

    fit!(model, x, y)

    # Accuracy testing

    accuracy = minimum(cross_val_score(model, x, y, cv = 5)) 
    return accuracy
end

# Making predictions and saving them into a CSV in the
# desired Kaggle format:

function CSV_data()
    df_test_ = df_test_creation()
    survived = ScikitLearn.predict(model, Matrix(df_test_))
    kaggle_df = DataFrame(PassengerId = 892:1309, Survived = survived)
    CSV.write("/Users/gabilio/Documents/titanic_kaggle/data/kaggle.csv", kaggle_df)
end