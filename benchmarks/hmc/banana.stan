// 2-D "banana" target, matched to the anvl / torch potential energy
//   U(theta) = theta[1]^2 / 200 + (theta[2] - b * theta[1]^2 + 100 * b)^2 / 2
// (the model block accumulates the log-density, i.e. -U).
data {
  real b;
}
parameters {
  vector[2] theta;
}
model {
  target += -square(theta[1]) / 200
            - square(theta[2] - b * square(theta[1]) + 100 * b) / 2;
}
