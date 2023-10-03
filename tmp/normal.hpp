// Code generated by stanc v2.33.0
#include <stan/model/model_header.hpp>
namespace normal_model_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 13> locations_array__ =
  {" (found before start of program)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 11, column 2 to column 24)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 12, column 2 to column 27)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 31, column 4 to column 47)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 19, column 2 to column 40)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 20, column 2 to column 41)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 24, column 4 to column 88)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 23, column 17 to line 25, column 3)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 23, column 2 to line 25, column 3)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 3, column 2 to column 20)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 4, column 8 to column 9)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 4, column 2 to column 26)",
  " (in '/Users/ddifrancesco/Github/IMAC_XLII/tmp/normal.stan', line 5, column 2 to column 27)"};
class normal_model final : public model_base_crtp<normal_model> {
 private:
  int N;
  std::vector<double> test_data;
  double epsilon;
 public:
  ~normal_model() {}
  normal_model(stan::io::var_context& context__, unsigned int
               random_seed__ = 0, std::ostream* pstream__ = nullptr)
      : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double;
    boost::ecuyer1988 base_rng__ =
      stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ =
      "normal_model_namespace::normal_model";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 9;
      context__.validate_dims("data initialization", "N", "int",
        std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      current_statement__ = 9;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 9;
      stan::math::check_greater_or_equal(function__, "N", N, 0);
      current_statement__ = 10;
      stan::math::validate_non_negative_index("test_data", "N", N);
      current_statement__ = 11;
      context__.validate_dims("data initialization", "test_data", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      test_data = std::vector<double>(N,
                    std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 11;
      test_data = context__.vals_r("test_data");
      current_statement__ = 12;
      context__.validate_dims("data initialization", "epsilon", "double",
        std::vector<size_t>{});
      epsilon = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 12;
      epsilon = context__.vals_r("epsilon")[(1 - 1)];
      current_statement__ = 12;
      stan::math::check_greater_or_equal(function__, "epsilon", epsilon, 0);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + 1;
  }
  inline std::string model_name() const final {
    return "normal_model";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.33.0",
             "stancflags = "};
  }
  // Base log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_not_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "normal_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      local_scalar_t__ mu_k = DUMMY_VAR__;
      current_statement__ = 1;
      mu_k = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
               lp__);
      local_scalar_t__ sigma_k = DUMMY_VAR__;
      current_statement__ = 2;
      sigma_k = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      {
        current_statement__ = 4;
        lp_accum__.add(stan::math::normal_lpdf<false>(mu_k, 200, 50));
        current_statement__ = 5;
        lp_accum__.add(stan::math::normal_lpdf<false>(sigma_k, 0, 30));
        current_statement__ = 8;
        for (int i = 1; i <= N; ++i) {
          current_statement__ = 6;
          lp_accum__.add(stan::math::normal_lpdf<false>(
                           stan::model::rvalue(test_data, "test_data",
                             stan::model::index_uni(i)), mu_k,
                           stan::math::sqrt((stan::math::square(epsilon) +
                             stan::math::square(sigma_k)))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  // Reverse mode autodiff log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "normal_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      local_scalar_t__ mu_k = DUMMY_VAR__;
      current_statement__ = 1;
      mu_k = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
               lp__);
      local_scalar_t__ sigma_k = DUMMY_VAR__;
      current_statement__ = 2;
      sigma_k = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      {
        current_statement__ = 4;
        lp_accum__.add(stan::math::normal_lpdf<false>(mu_k, 200, 50));
        current_statement__ = 5;
        lp_accum__.add(stan::math::normal_lpdf<false>(sigma_k, 0, 30));
        current_statement__ = 8;
        for (int i = 1; i <= N; ++i) {
          current_statement__ = 6;
          lp_accum__.add(stan::math::normal_lpdf<false>(
                           stan::model::rvalue(test_data, "test_data",
                             stan::model::index_uni(i)), mu_k,
                           stan::math::sqrt((stan::math::square(epsilon) +
                             stan::math::square(sigma_k)))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    // suppress unused var warning
    (void) jacobian__;
    static constexpr const char* function__ =
      "normal_model_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      double mu_k = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      mu_k = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
               lp__);
      double sigma_k = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 2;
      sigma_k = in__.template read_constrain_lb<local_scalar_t__,
                  jacobian__>(0, lp__);
      out__.write(mu_k);
      out__.write(sigma_k);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
      double Kmat_pred = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      Kmat_pred = stan::math::normal_rng(mu_k, sigma_k, base_rng__);
      out__.write(Kmat_pred);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      local_scalar_t__ mu_k = DUMMY_VAR__;
      current_statement__ = 1;
      mu_k = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, mu_k);
      local_scalar_t__ sigma_k = DUMMY_VAR__;
      current_statement__ = 2;
      sigma_k = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma_k);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "mu_k", "double",
        std::vector<size_t>{});
      current_statement__ = 2;
      context__.validate_dims("parameter initialization", "sigma_k",
        "double", std::vector<size_t>{});
      local_scalar_t__ mu_k = DUMMY_VAR__;
      current_statement__ = 1;
      mu_k = context__.vals_r("mu_k")[(1 - 1)];
      out__.write_free_lb(0, mu_k);
      local_scalar_t__ sigma_k = DUMMY_VAR__;
      current_statement__ = 2;
      sigma_k = context__.vals_r("sigma_k")[(1 - 1)];
      out__.write_free_lb(0, sigma_k);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"mu_k", "sigma_k"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      std::vector<std::string> temp{"Kmat_pred"};
      names__.reserve(names__.size() + temp.size());
      names__.insert(names__.end(), temp.begin(), temp.end());
    }
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
                std::vector<size_t>{}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      std::vector<std::vector<size_t>> temp{std::vector<size_t>{}};
      dimss__.reserve(dimss__.size() + temp.size());
      dimss__.insert(dimss__.end(), temp.begin(), temp.end());
    }
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "mu_k");
    param_names__.emplace_back(std::string() + "sigma_k");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      param_names__.emplace_back(std::string() + "Kmat_pred");
    }
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    param_names__.emplace_back(std::string() + "mu_k");
    param_names__.emplace_back(std::string() + "sigma_k");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      param_names__.emplace_back(std::string() + "Kmat_pred");
    }
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"mu_k\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma_k\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"Kmat_pred\",\"type\":{\"name\":\"real\"},\"block\":\"generated_quantities\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"mu_k\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma_k\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"Kmat_pred\",\"type\":{\"name\":\"real\"},\"block\":\"generated_quantities\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (1 + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (1);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (1 + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (1);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = normal_model_namespace::normal_model;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return normal_model_namespace::profiles__;
}
#endif