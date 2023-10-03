functions {
   
  real normal_lub_rng(real mu, real sigma, real lb, real ub) {
    real p_lb = normal_cdf(lb | mu, sigma);
    real p_ub = normal_cdf(ub | mu, sigma);
    real u = uniform_rng(p_lb, p_ub);
    real y = mu + sigma * inv_Phi(u);
    return y; 
  }
   
}

data {

  int <lower = 0> N;  // Number of data points
  array[N] real test_data;  // Observed data
  real <lower = 0> epsilon;  // Known standard deviation for the likelihood

}

parameters {

  real <lower = 0> mu_k;  // Mean
  real <lower = 0> sigma_k;  // Standard deviation

}

model {

  // Priors
  target += normal_lpdf(mu_k | 200, 50);
  target += normal_lpdf(sigma_k | 0, 30);

  // Likelihood
  for (i in 1:N) {
    target += normal_lpdf(test_data[i] | mu_k, sqrt(square(epsilon) + square(sigma_k)));
  }

}

generated quantities {
     
    real Kmat_pred = normal_lub_rng(mu_k, sigma_k, 0, positive_infinity());

}