data {
  int <lower = 0> N;  // number of data points
  vector [N] CVN_J;  // data for CVN_ft_lbs
  vector [N] Kmat_MPa_m;  // observed output data
  real CVN_J_pred;  // CVN_ft_lbs prediction
}

parameters {
  real <lower = 0> sigma;  // model uncertainty (noise) standard deviation
}

model {
  vector [N] mu;
  
  // Priors
  target += exponential_lpdf(sigma | 1);
  
  // Likelihood
  for (n in 1:N) {
    mu[n] = 0.54 * CVN_J[n] + 55;
    target += normal_lpdf(Kmat_MPa_m[n] | mu[n], sigma);
  }
  
}

generated quantities {
    real Kmat_MPa_m_pred;
    real mu_pred = 0.54 * CVN_J_pred + 55;
    
    Kmat_MPa_m_pred= normal_rng(mu_pred, sigma);
}
