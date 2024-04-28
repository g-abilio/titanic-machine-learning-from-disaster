# Adding modules
using Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Distributions")

# Import modules
using DataFrames
using CSV
using Distributions

# After studies on this DataFrame, certain operations
# will be done, with the aim to get better results in
# the ML model training

function df_train_creation()
    # Uploading training data 
    
    df_train = CSV.read("/Users/gabilio/Documents/titanic_kaggle/data/train.csv", DataFrame)

    # Embarked has 2 missing rows... As it is only 2, it
    # can be dropped with no big problems
   
    df_train = dropmissing(df_train, :Embarked)

    # Now, we are going to fill the missings in the Age
    # column with a normal distribution. The mean and std 
    # were calculated previously

    new_age = round.(Int64, rand(Distributions.Normal(29.64209269662921, 14.49293290032352), 177))

    # Removing the negative values

    for i in 1:length(new_age)
        if new_age[i] < 0
            new_age[i] = 1
        end
    end

    # Inserting the values on the df

    global iterator_j = 1

    for i in 1:length(df_train.Age)
        if ismissing(df_train.Age[i])
            df_train.Age[i] = new_age[iterator_j]
            global iterator_j += 1
        end
    end

    # As the Cabin column has a big number of missing
    # values and it is not as important for the learning,
    # it will be dropped

    df_train = select(df_train, Not("Cabin"))

    # Name column will be dropped because it is not that
    # important for the learning phase

    df_train = select(df_train, Not("Name"))

    # Now, let's make the non-numeric columns numerical:

    df_train.Embarked = Int64.(replace(df_train.Embarked, "S" => 0, "C" => 1, "Q" => 2))
    df_train.Sex = Int64.(replace(df_train.Sex, "male" => 0, "female" => 1))

    # Ticket column will be dropped, as it has 680 different
    # values and it is not relevant for learning. The same 
    # will be happening with PassengerId

    df_train = select(df_train, Not("Ticket"))
    df_train = select(df_train, Not("PassengerId"))

    # Creating Family size column, that is relevant in
    # this cenario

    df_train[:, :Family_size] = df_train[:, :SibSp] .+ df_train[:, :Parch]

    # Creating a column to depict if a person was alone.
    # Another relevant topic in this case

    is_alone = []

    for i in df_train.Family_size
        if i > 0
            append!(is_alone, 0)
        else
            append!(is_alone, 1)
        end
    end

    df_train.alone = Int64.(is_alone)
    return df_train
end
