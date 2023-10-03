data {
  int <lower = 0> N;  // number of data points
  vector [N] CVN_ft_lbs;  // data for CVN_ft_lbs
  vector [N] sigmaY_ksi;  // data for sigmaY_ksi
  vector [N] Kmat_ksi_in;  // observed output data
  real CVN_ft_lbs_pred;  // CVN_ft_lbs prediction
  real sigmaY_ksi_pred;  // sigmaY_ksi prediction
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
    mu[n] = sqrt(5 * CVN_ft_lbs[n] * sigmaY_ksi[n] - (sigmaY_ksi[n]^2)/4);
    target += normal_lpdf(Kmat_ksi_in[n] | mu[n], sigma);
  }
  
}

generated quantities {
    real Kmat_pred_ksi;
    real mu_pred = sqrt(5 * CVN_ft_lbs_pred * sigmaY_ksi_pred - (sigmaY_ksi_pred^2)/4);
    
    Kmat_pred_ksi= normal_rng(mu_pred, sigma);
}