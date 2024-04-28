# Adding modules
using Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Distributions")

# Import modules
using DataFrames
using CSV
using Distributions

# Now, we are going to apply the operations on the
# training set to the test set

function df_test_creation()
    # Uploading test data 
    
    df_test = CSV.read("/Users/gabilio/Documents/titanic_kaggle/data/test.csv", DataFrame)

    # Filling the misses in the age column with Gaussian
    # distribution. The mean and std were calculated previously

    new_age = round.(Int64, rand(Distributions.Normal(30.272590361445783, 14.181209235624422), 86))

    for i in 1:length(new_age)
        if new_age[i] < 0
            new_age[i] = 1
        end
    end

    global j = 1

    for i in 1:length(df_test.Age)
        if ismissing(df_test.Age[i])
            df_test.Age[i] = new_age[j]
            global j += 1
        end
    end

    # Filling the missing value in the Fare column with
    # the mean value

    df_test.Fare = replace(df_test.Fare, missing => 35.6272)

    # Applying all the operations of the train set into
    # the test set

    df_test = select(df_test, Not("Cabin"))
    df_test = select(df_test, Not("Name"))
    df_test.Embarked = Int64.(replace(df_test.Embarked, "S" => 0, "C" => 1, "Q" => 2))
    df_test.Sex = Int64.(replace(df_test.Sex, "male" => 0, "female" => 1))
    df_test = select(df_test, Not("Ticket"))
    df_test = select(df_test, Not("PassengerId"))

    # Creation of Family_size and alone columns

    df_test[:, :Family_size] = df_test[:, :SibSp] .+ df_test[:, :Parch]

    is_alone = []
    for i in df_test.Family_size
        if i > 0
            append!(is_alone, 0)
        else
            append!(is_alone, 1)
        end
    end

    df_test.alone = Int64.(is_alone)

    return df_test
end