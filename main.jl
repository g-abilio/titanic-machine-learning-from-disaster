# Import modules

using Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Distributions")
Pkg.add("DecisionTree")
Pkg.add("ScikitLearn")

using DataFrames
using CSV
using Distributions
using DecisionTree
using ScikitLearn
using ScikitLearn.CrossValidation

include("libs/train_df_creation.jl")
include("libs/test_df_creation.jl")
include("libs/ml_model.jl")

# Main scope

println("*** Titanic - Machine Learning from Disaster ***")
println("An approach using Julia")

println("It was used different tools to make the dataset propicious to better learning")
println("Do you want to see the train and test datasets? yes or no?")

question = readline()

if question == "yes"

    # Train data 

    df_train = df_train_creation()
    println("Train DataFrame:")
    println(df_train)

    # Test data

    df_test = df_test_creation()
    println("Test DataFrame:")
    println(df_test)

elseif question == "no"
    println("Ok!")

else 
    println("Invalid answer... Anyway, let's go to the results!")

end

println("The ML model used was the RandomForestClassifier, and the accuracy obtained was:")
accuracy = model_development()
println(accuracy)

# Saving the predictions into a CSV

CSV_data()