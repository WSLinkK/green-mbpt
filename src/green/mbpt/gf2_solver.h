/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef MPIGF2_DFGF2SOLVER_H
#define MPIGF2_DFGF2SOLVER_H

#include <green/grids/transformer_t.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <mpi.h>

#include <Eigen/Core>

#include "common_defs.h"
#include "df_integral_t.h"

namespace green::mbpt {
  /**
   * @brief This class performs self-energy calculation by means of second-order PT using density fitting
   */
  class gf2_solver {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using S1_type    = ztensor<4>;
    using St_type    = utils::shared_object<ztensor<5>>;

  public:

    /**
     *
     * @param p - command line parameters
     * @param tr - time-frequency transform
     * @param bz - Brillouin transform
     */
    gf2_solver(const params::params& p, const grids::transformer_t& tr, const bz_utils_t& bz) :
        _nts(tr.sd().repn_fermi().nts()), _nk(bz.nk()), _ink(bz.ink()), _path(p["dfintegral_file"]),
        _ewald(std::filesystem::exists(_path + "/df_ewald.h5")), _bz_utils(bz),
        statistics("GF2") {
      h5pp::archive ar(p["input_file"]);
      ar["params/nao"] >> _nao;
      ar["params/nso"] >> _nso;
      ar["params/ns"] >> _ns;
      ar["params/NQ"] >> _NQ;
      ar.close();

      h5pp::archive core(p["atom_core_file"]);
      core[p["atom_store_key"]] >> _core_sigma;
      core.close();
      
      const std::array<size_t,5> &shp = _core_sigma.shape();
      assert(shp[0] == _nts);
      assert(shp[1] == _ns);
      assert(shp[2] == _nk);
      assert(shp[3] == _nao);
      assert(shp[4] == _nao);

      std::cout << _core_sigma(0,0,0,0,0) << std::endl;
      const std::string valence_rows_str = p["valence_rows"];
      const std::string valence_cols_str = p["valence_cols"];

      _valence_rows = parse_index_list(valence_rows_str);

      if (!valence_cols_str.empty()) {
        _valence_cols = parse_index_list(valence_cols_str);
      } else {
        _valence_cols = _valence_rows;  // default: same as rows
      }
      _frozen_core_mode = p["frozen_core_mode"];
    }

     /**
      * Solve GF2 equations for Self-energy
      *
      * @param g_tau - Green's function object
      * @param sigma1 - static part of the self-energy
      * @param sigma_tau - dynamical part of the self-energy
      */
    void solve(G_type& g_tau, S1_type& sigma1, St_type& sigma_tau);

  private:
    // dimension of problem (nao*ncell)
    size_t            _dim;
    // number of time steps
    size_t            _nts;

    size_t            _nk;
    size_t            _ink;
    size_t            _nao;
    size_t            _nso;
    size_t            _ns;
    size_t            _NQ;

    std::vector<std::size_t> _valence_rows;
    std::vector<std::size_t> _valence_cols;

    frozen_core_mode_e       _frozen_core_mode;

    // Path to H5 file
    const std::string _path;

    // references to arrays
    ztensor<5>        Sigma_local;
    ztensor<5>        _core_sigma;

    // Current time step Green's function matrix for k1
    Eigen::MatrixXcd  _G_k1_tmp;
    // Current reverse time step Green's function matrix for k2
    Eigen::MatrixXcd  _Gb_k2_tmp;
    // Current time step Green's function matrix for k3
    Eigen::MatrixXcd  _G_k3_tmp;

    /**
     * Read next part of Coulomb integrals for fixed set of k-points
     */
    void              read_next(const std::array<size_t, 4>& k);

    // Compute correction into second-order from the divergent G=0 part of the interaction
    void              compute_2nd_exch_correction(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau);

    void      ewald_2nd_order_0_0(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3, MMatrixXcd& Xm_4,
                                  MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, MMatrixXcd& Xm,
                                  MMatrixXcd& Vm);

    void      ewald_2nd_order_1_0(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3, MMatrixXcd& Xm_4,
                                  MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, MMatrixXcd& Xm,
                                  MMatrixXcd& Vm);

    void      ewald_2nd_order_0_1(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3, MMatrixXcd& Xm_4,
                                  MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, MMatrixXcd& Xm,
                                  MMatrixXcd& Vm);

    // Read next portion of the correction to a coulomb integral
    void      read_next_correction_0_0(size_t k);

    void      read_next_correction_1_0(size_t k1, size_t k2);

    void      read_next_correction_0_1(size_t k1, size_t k2);

    /**
     * Performs loop over time for fixed set of k-points
     */
    void      selfenergy_innerloop(size_t tau_offset, size_t ntau_local, const std::array<size_t, 4>& k, size_t is, const ztensor<5>& Gr_full_tau);

    MatrixXcd extract_G_tau_k(const ztensor<5>& G_tau, size_t t, size_t k_pos, size_t k_red, size_t s) {
      int         ts_shift = t * G_tau.shape()[1] * G_tau.shape()[2] * _nao * _nao + s * G_tau.shape()[2] * _nao * _nao;
      int         k_shift  = k_pos * _nao * _nao;
      CMMatrixXcd tmp(G_tau.data() + ts_shift + k_shift, G_tau.shape()[3], G_tau.shape()[4]);
      MatrixXcd   G = tmp;
      if (_bz_utils.symmetry().conj_list()[k_red] != 0) {
        for (size_t i = 0; i < _nao; ++i) {
          for (size_t j = 0; j < _nao; ++j) {
            G(i, j) = std::conj(G(i, j));
          }
        }
      }

      return G;
    }

    //* --- parse core_rows / core_cols from params ---
    std::vector<std::size_t> parse_index_list(const std::string& s) const {
      std::vector<std::size_t> idx;
      std::stringstream ss(s);
      std::string token;

      while (std::getline(ss, token, ',')) {
        // trim spaces
        std::size_t begin = token.find_first_not_of(" \t");
        if (begin == std::string::npos) {
          continue;
        }
        std::size_t end = token.find_last_not_of(" \t");
        std::string trimmed = token.substr(begin, end - begin + 1);

        if (!trimmed.empty()) {
          std::size_t value = static_cast<std::size_t>(std::stoul(trimmed));
          idx.push_back(value);
        }
      }

      return idx;
    }

    Eigen::MatrixXcd restrict_orbitals(
    const Eigen::MatrixXcd& input,
    const std::vector<std::size_t>& idx_row,
    const std::vector<std::size_t>& idx_col) const
    {
        Eigen::MatrixXcd output = Eigen::MatrixXcd::Zero(input.rows(), input.cols());

        for (std::size_t a = 0; a < idx_row.size(); ++a) {
            std::size_t i = idx_row[a];
            for (std::size_t b = 0; b < idx_col.size(); ++b) {
                std::size_t j = idx_col[b];
                output(i, j) = input(i, j);
            }
        }
        return output;
    }

    /**
     * Performs all possible contractions for i and n indices
     */
    void              contraction(size_t nao2, size_t nao3, bool eq_spin, bool ew_correct, const Eigen::MatrixXcd& G1,
                                  const Eigen::MatrixXcd& G2, const Eigen::MatrixXcd& G3, MMatrixXcd& Xm_4, MMatrixXcd& Xm_1, MMatrixXcd& Xm_2,
                                  MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, const MMatrixXcd& vm_1, MMatrixXcd& Xm, MMatrixXcd& Vm, MMatrixXcd& Vxm,
                                  MMatrixXcd& Sm);

    /**
     * Compute two-electron integrals for the fixed set of k-points using pre-computed fitted densities
     *
     * @param set of k-points
     */
    void              setup_integrals(const std::array<size_t, 4>& k);

    // Pre-computed fitted densities
    // To avoid divergence in G=0 we separately compute ewald correction for the divergent part
    // Left interaction term
    df_integral_t*    _coul_int_c_1;
    df_integral_t*    _coul_int_c_2;
    // Right direct term
    df_integral_t*    _coul_int_c_3;
    df_integral_t*    _coul_int_c_4;
    // Right exchange term
    df_integral_t*    _coul_int_x_3;
    df_integral_t*    _coul_int_x_4;

    ztensor<4>        vijkl;
    ztensor<4>        vcijkl;
    ztensor<4>        vxijkl;
    ztensor<4>        vxcijkl;

    bool              _ewald;

    const bz_utils_t& _bz_utils;

    //
    utils::timing     statistics;
  };
}  // namespace green::mbpt

#endif  // MPIGF2_DFGF2SOLVER_H
