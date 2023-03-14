# Building your Diet using Artificial Intelligence

_This model does not replace a certified dietician and is merely a simplified approach to a complex problem._

The math behind this project is linear optimization, specifically the simplex algorithm. The goal is to minimize the number of calories while still satisfying the nutritional requirements for proteins, carbohydrates, and fat.

To formalize the problem, we represent the amount of each food item consumed as a vector x. We also have vectors f, e, and p representing the amount of fat, carbohydrates, and protein per 100 grams of each food item. Finally, we have constraints on the total amount of fat, carbohydrates, and protein that need to be consumed.

We can represent the problem as a linear programming problem:

`minimize c^T x`

`subject to Ax = b`

`x >= 0`

Here, c is the vector of calorie values for each food item, A is the matrix of f, e, and p values for each food item, and b is the vector of required amounts of fat, carbohydrates, and protein. The objective is to minimize the total number of calories consumed.

The simplex algorithm is used to solve this linear programming problem. The algorithm starts at a feasible solution and iteratively moves to adjacent feasible solutions that improve the objective function until it reaches the optimal solution. The optimal solution is the set of values for x that minimize the objective function while satisfying the constraints.

In this project, the simplex algorithm is implemented using the PuLP library in Python. The result is a diet plan that satisfies the nutritional requirements while minimizing the number of calories consumed.

## Architecture Overview
We will train model locally, upload to Google Cloud, invoke Cloud Functions via API request to download model and run linear optimization:

![](resources/architecture.png)