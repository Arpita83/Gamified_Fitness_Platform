# code to calculate calories burnt during workout
# no special packages required
# creating function
def countcals(gender, age, weight, heart_rate, time):
    # determining the gender
    if gender == "male":
        # implementing calculation for male
        calories = (
            ((age * 0.2017) + (weight * 0.09036) + (heart_rate * 0.6309) - 55.0969)
            * time
        ) / 4.184
    elif gender == "female":
        # implementing calculation for female
        calories = (
            ((age * 0.074) + (weight * 0.05741) + (heart_rate * 0.4472) - 20.4022)
            * time
        ) / 4.184
    else:
        # error raised
        calories = "Cannot be calculated, wrong input!"
    # function will return calories burnt based on the formulae
    return calories
 
 
# driver code
# getting user input
gender = input("Enter gender,(male/female)= ")
age = int(input("Enter age in years only= "))
weight = int(input("Enter weight in kilograms only= "))
heart_rate = int(input("Enter heart rate= "))
time = int(input("Enter duration of workout in hours only= "))
# function call
burnedcals = countcals(gender, age, weight, heart_rate, time)
# display result
print("the calories you burnt were= ", burnedcals)
